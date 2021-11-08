"""
Functions used in WGAN are based on the keras official document https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py
Thanks for them!
"""
import argparse
import os
from functools import partial

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten, Input, Reshape
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K 
import cv2
from tqdm import tqdm
import copy
import math

import MnasNet
import model
from Utils import train_same_people_DataSequence


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from skimage.measure import compare_ssim
import seaborn as sns




args_parser = argparse.ArgumentParser()



args_parser.add_argument("--train-dir", type=str, required=False, help="Directory of real dataset for train.")
args_parser.add_argument("--real-dir", type=str, required=False, help="Directory of real dataset for test.")
args_parser.add_argument("--fake-dir", type=str, required=False, help="Directory of fake dataset for test.")
args_parser.add_argument("--faceswap-dir", type=str, required=False, help="Directory of real dataset.")
args_parser.add_argument("--gan-dir", type=str, required=False, help="Directory of real dataset.")
args_parser.add_argument("--lip-dir", type=str, required=False, help="Directory of real dataset.")
args_parser.add_argument("--model-name", type=str, required=True, help="name of model.")

args_parser.add_argument("-p", "--pretrain-dir", type=str, required=False,
                         help="(Optional) Selection of testing GPU.")

args_parser.add_argument("-b", "--batch-size", type=int, required=True, help="number of batch size.")
args_parser.add_argument("-g", "--gpu-num", type=str, required=False, default=0,
                         help="(Optional) Selection of testing GPU.")
args_parser.add_argument("--opr", type=str, required=True,choices=["train","test"],
                         help="(Optional) Selection of operation, including train and test.")

args_parser.add_argument("--pretrain-model", type=str, required=False,
                         help="(Optional) Directory of pretrain-model.")
args_parser.add_argument("-l","--learning-rate", type=float, required=False, default=1e-5,
                         help="(Optional) Set the learning rate, default 1e-5.")
args_parser.add_argument("-e","--epoches", type=int, required=False, default=5,
                         help="(Optional) Set the number of epoch, default 5.")
args_parser.add_argument("--person", type=str, required=True, default='01',
                         help="Selection of person.")
args_parser.add_argument("--famous", action="store_true",
                         help="Famous or not.") 
args_parser.add_argument("--save-path", type=str, required=False, help="the path which save the result.")
args_parser.add_argument("--use-gan", action="store_true", help="If you set this, the enhance model will be choosen as WGAN, default Perceptual loss model.")
args_parser.add_argument("--lenth", type=int, required=True, default=1024, help="name of csv.")
args = args_parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num


# tf_config = tf.ConfigProto()  
# tf_config.gpu_options.allow_growth = True
# session = tf.Session(config=tf_config) 

BATCH_SIZE = args.batch_size
# The training ratio is the number of discriminator updates
# per generator update. The paper uses 5.
TRAINING_RATIO = 5
GRADIENT_PENALTY_WEIGHT = 10  # As per the paper
style_len = int(args.lenth/2)

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


def gradient_penalty_loss(y_true, y_pred, averaged_samples,
                          gradient_penalty_weight):
    # first get the gradients:
    #   assuming: - that y_pred has dimensions (batch_size, 1)
    #             - averaged_samples has dimensions (batch_size, nbr_features)
    # gradients afterwards has dimension (batch_size, nbr_features), basically
    # a list of nbr_features-dimensional gradient vectors
    gradients = K.gradients(y_pred, averaged_samples)[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)


def make_generator():
    Model_content = MnasNet.MnasNet(n_classes=512)
    Model_generator = model.decoder(content_len=args.content)
    input_face_mini = tf.keras.layers.Input(shape=(32,32,3))
    input_face = tf.keras.layers.Input(shape=(256,256,3))

    vec_content = Model_content(input_face)
    output_face = Model_generator([input_face_mini,vec_content])

    return tf.keras.models.Model(inputs=[input_face_mini,input_face],outputs=[output_face,vec_content])


def make_discriminator():
    def d_layer(layer_input, filters, f_size=4, normalization=True):
        d = tf.keras.layers.Conv2D(filters, kernel_size=f_size, strides=2, padding='same',kernel_initializer='he_normal')(layer_input)
        d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)
        # if normalization:
            # d = InstanceNormalization()(d)
        return d
    df=64

    img = tf.keras.layers.Input(shape=(256,256,3))
    d1 = d_layer(img, df, normalization=False)
    d2 = d_layer(d1, df*2)
    d3 = d_layer(d2, df*4)
    d4 = d_layer(d3, df*8)
    validity = tf.keras.layers.Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)
    dense1 = tf.keras.layers.LeakyReLU(alpha=0.2)(validity)
    dense1 = tf.keras.layers.Flatten()(dense1)
    dense1 = tf.keras.layers.Dense(1, kernel_initializer='he_normal')(dense1)
    return tf.keras.models.Model(inputs = img, outputs = dense1)



def tile_images(image_stack):
    """Given a stacked tensor of images, reshapes them into a horizontal tiling for
    display."""
    assert len(image_stack.shape) == 3
    image_list = [image_stack[i, :, :] for i in range(image_stack.shape[0])]
    tiled_images = np.concatenate(image_list, axis=1)
    return tiled_images


def RandomWeightedAverage(inputs):
    """Takes a randomly-weighted average of two tensors. In geometric terms, this
    outputs a random point on the line between each pair of input points.
    Inheriting from _Merge is a little messy but it was the quickest solution I could
    think of. Improvements appreciated."""

    weights = K.random_uniform((BATCH_SIZE, 1, 1, 1))
    return (weights * inputs[0]) + ((1 - weights) * inputs[1])

def my_loss(y_true,y_pred):
    return abs(y_pred)

def Gram_loss(y_true,y_pred):
    print(y_pred.shape)
    #tf_session = K.get_session()

    #y_pred_re = tf.reshape(y_pred,[shape_y_pred[0],shape_y_pred[1]*shape_y_pred[2],shape_y_pred[3]])
    #y_true_re = tf.reshape(y_true,[shape_y_true[0],shape_y_true[1]*shape_y_true[2],shape_y_true[3]])
    shape_y_true = K.int_shape(y_pred)   
    y_pred_l = tf.matmul(y_pred,y_pred,transpose_a=True)
    y_true_l = tf.matmul(y_true,y_true,transpose_a=True)
    y_pred_lnor = tf.divide(y_pred_l,shape_y_true[1]*shape_y_true[2])
    y_true_lnor = tf.divide(y_true_l,shape_y_true[1]*shape_y_true[2])
    loss = tf.subtract(y_pred_lnor,y_true_lnor)
    #K.int_shape(y_pred)

    n_loss = tf.nn.l2_loss(loss)
    nor_loss = tf.sqrt(n_loss)
    #nor_loss = tf.divide(n_loss,shape_y_true[1]*shape_y_true[2])
    return nor_loss


def cos_distance(y_true, y_pred):
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return K.mean(1 - K.sum((y_true * y_pred), axis=-1))

generator,style_model,content_model,re_model = model.reconstruction_model(content_len=int(style_len*2))


generator.compile(optimizer=Adam(args.learning_rate, beta_1=0.5, beta_2=0.9),
                        loss=['mse','mse']+[my_loss for i in range(6)],loss_weights=[10,100]+[1 for i in range(6)])

generator.trainable = True
generator_input = tf.keras.layers.Input(shape=(256,256,3))
style_input =  tf.keras.layers.Input(shape=(256,256,3))

xception_model = tf.keras.applications.xception.Xception(include_top=False, weights='imagenet', input_tensor=None, input_shape=(256,256,3), pooling=None, classes=1000)

if args.use_gan:
    discriminator = make_discriminator()

    # Now that the generator_model is compiled, we can make the discriminator
    # layers trainable.
    # for layer in discriminator.layers:
    #     layer.trainable = True
    # for layer in generator.layers:
    #     layer.trainable = False
    discriminator.trainable = True
    generator.trainable = False

    # The discriminator_model is more complex. It takes both real image samples and random
    # noise seeds as input. The noise seed is run through the generator model to get
    # generated images. Both real and generated images are then run through the
    # discriminator. Although we could concatenate the real and generated images into a
    # single tensor, we don't (see model compilation for why).
    real_samples = Input(shape=(256,256,3))
    generator_input_for_discriminator = Input(shape=(256,256,3))
    generator_style_for_discriminator = Input(shape=(256,256,3))

    generated_samples_for_discriminator,oc_vec,dot_vec = generator([generator_style_for_discriminator,generator_input_for_discriminator])
    discriminator_output_from_generator = discriminator(generated_samples_for_discriminator)
    discriminator_output_from_real_samples = discriminator(real_samples)

    # We also need to generate weighted-averages of real and generated samples,
    # to use for the gradient norm penalty.
    averaged_samples = tf.keras.layers.Lambda(RandomWeightedAverage)([real_samples,
                                                generated_samples_for_discriminator])
    # We then run these samples through the discriminator as well. Note that we never
    # really use the discriminator output for these samples - we're only running them to
    # get the gradient norm for the gradient penalty loss.
    averaged_samples_out = discriminator(averaged_samples)

    # The gradient penalty loss function requires the input averaged samples to get
    # gradients. However, Keras loss functions can only have two arguments, y_true and
    # y_pred. We get around this by making a partial() of the function with the averaged
    # samples here.
    partial_gp_loss = partial(gradient_penalty_loss,
                            averaged_samples=averaged_samples,
                            gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT)
    # Functions need names or Keras will throw an error
    partial_gp_loss.__name__ = 'gradient_penalty'

    # Keras requires that inputs and outputs have the same number of samples. This is why
    # we didn't concatenate the real samples and generated samples before passing them to
    # the discriminator: If we had, it would create an output with 2 * BATCH_SIZE samples,
    # while the output of the "averaged" samples for gradient penalty
    # would have only BATCH_SIZE samples.

    # If we don't concatenate the real and generated samples, however, we get three
    # outputs: One of the generated samples, one of the real samples, and one of the
    # averaged samples, all of size BATCH_SIZE. This works neatly!
    discriminator_model = Model(inputs=[real_samples,generator_style_for_discriminator,
                                        generator_input_for_discriminator],
                                outputs=[discriminator_output_from_real_samples,
                                        discriminator_output_from_generator,
                                        averaged_samples_out])
    # We use the Adam paramaters from Gulrajani et al. We use the Wasserstein loss for both
    # the real and generated samples, and the gradient penalty loss for the averaged samples
    discriminator_model.compile(optimizer=Adam(args.learning_rate, beta_1=0.5, beta_2=0.9,decay=1e-5),
                                loss=[wasserstein_loss,
                                    wasserstein_loss,
                                    partial_gp_loss])

    discriminator.trainable = False
    generator.trainable = True


    generator_model = Model(inputs=[generator_style_for_discriminator,generator_input_for_discriminator],
                            outputs=[generated_samples_for_discriminator,oc_vec,dot_vec,discriminator_output_from_generator])

    generator_model.compile(optimizer=Adam(args.learning_rate, beta_1=0.5, beta_2=0.9),
                            loss=['mse','mse',my_loss,wasserstein_loss],loss_weights=[1,1,1,1])



else:

    test_model = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=(256,256,3), pooling=None, classes=1000)
    test_model.summary()
    test_model.trainable = False




    selectedLayers = [
        'block1_conv2',
        'block2_conv2',
        'block3_conv3',
        'block4_conv3'
    ]
    selectedstyleLayers = [
        'block1_conv2'
    ]
    selectedcontentLayers = [
        'block2_conv2'
    ]
    test_layers = [test_model.get_layer(i) for i in selectedLayers]
    layer_shape = [K.int_shape(test_layer.output) for test_layer in test_layers]
    reshape_layers = [tf.keras.layers.Reshape((layer_shape[i][1]*layer_shape[i][2],layer_shape[i][3]))(test_layer.output) for i,test_layer in enumerate(test_layers)]

    #perceptual = tf.keras.models.Model(inputs=test_model.get_layer('input_8').output,outputs=reshape_layers)# [test_model.get_layer(i).output for i in selectedcontentLayers]
    style_preceptual = tf.keras.models.Model(inputs=test_model.get_layer('input_10').output,outputs=reshape_layers)
    content_perceptual = tf.keras.models.Model(inputs=test_model.get_layer('input_10').output,outputs=[test_model.get_layer(i).output for i in selectedcontentLayers])

    style_preceptual.trainable = False
    content_perceptual.trainable = False
    
    output_for_discriminator,output_vec,kl_2_1,kl_2_2,kl_3_1,kl_3_2,kl_4_1,kl_4_2 = generator([style_input,generator_input])
    transfer_layer = tf.keras.layers.Lambda(lambda x: (x+1)/2)(output_for_discriminator)

    #fake_perceptual = perceptual(transfer_layer)
    content_loss = content_perceptual(transfer_layer)
    style_loss = style_preceptual(transfer_layer)
    generator_perceptual = tf.keras.models.Model(inputs=[style_input,generator_input],
                            outputs=[output_for_discriminator,output_vec,kl_2_1,kl_2_2,kl_3_1,kl_3_2,kl_4_1,kl_4_2,content_loss]+style_loss)


    generator_perceptual.compile(optimizer=Adam(args.learning_rate, beta_1=0.5, beta_2=0.9),
                            loss=['mse','mse']+[my_loss for i in range(6)]+['mse']+[Gram_loss for i in range(len(selectedLayers))],#['mse' for i in range(len(selectedcontentLayers))]
                            loss_weights=[100,100]+[5,5,5,5,5,5,0.1]+[0.01*(2^(i+1)) for i in range(len(selectedLayers))])





style_model.compile(optimizer=Adam(args.learning_rate),loss='mse')

positive_y = np.ones((BATCH_SIZE, 1), dtype=np.float)
negative_y = -positive_y
dummy_y = np.zeros((BATCH_SIZE, 1), dtype=np.float)




if args.pretrain_model is not None:
    generator.load_weights(os.path.join('./Model',args.pretrain_model+'_OC.hdf5'))
    
    if args.use_gan:
        discriminator_model.load_weights(os.path.join('./Model',args.model_name+'_OC_WGAN_D.hdf5'))


def train():
    best_g = 5
    for epoch in range(args.epoches):

        temp_g=0
        X_train = train_same_people_DataSequence(path=args.train_dir,batch_size=BATCH_SIZE,number=args.person,famous=args.famous)

        progress_bar = tf.keras.utils.Progbar(target=len(X_train))

        print('epoch ',epoch+1)

        g_loss=0
        gd=0
        dot_loss=0
        oc_loss=0
        d_l=0
        gen_loss=0
        for i in range(len(X_train)):
            image_batch = X_train[i]
            anchor = style_model.predict_on_batch(image_batch[0][1])
            

            
            if args.use_gan:
                #print(args.use_gan)
                discriminator.trainable = True
                generator.trainable = False

                for j in range(TRAINING_RATIO):
                    d_loss = discriminator_model.train_on_batch([image_batch[0][1], image_batch[1][1],image_batch[0][1]],[positive_y, negative_y, dummy_y])
                d_l = 0.5 * np.add(d_loss[1], d_loss[2])
                discriminator.trainable = False
                generator.trainable = True

                
                
                for k in range(1):
                    ge_loss = generator.train_on_batch([image_batch[1][1],image_batch[0][1]],[image_batch[0][1],anchor,positive_y])
                    #gd_loss = generator_dis.train_on_batch([image_batch[1][1],image_batch[0][1]],positive_y)
                    gd_loss = generator_model.train_on_batch([image_batch[1][1],image_batch[0][1]],[image_batch[0][1],anchor,positive_y,positive_y])
                    #print(generator_model.metrics_names)
                    g_loss=gd_loss[0]
                    gd=gd_loss[-1]
                    gen_loss = gd_loss[1]
                    dot_loss = gd_loss[3]
                    oc_loss = gd_loss[2]

            else:
                real_perceptual_style = style_preceptual.predict_on_batch(image_batch[0][1])
                real_perceptual_content = content_perceptual.predict_on_batch(image_batch[0][1])
                generator.trainable = True
                # if gen_loss>0.04:
            
                #ge_loss = generator.train_on_batch([image_batch[1][1],image_batch[0][1]],[image_batch[0][1],anchor]+[dummy_y for i in range(6)]) 
                ge_loss = generator_perceptual.train_on_batch([image_batch[1][1],image_batch[0][1]],[image_batch[0][1],anchor]+[dummy_y for i in range(6)]+[real_perceptual_content]+real_perceptual_style) 
                gd = ge_loss[-1]
                gen_loss = ge_loss[1] 
                g_loss = ge_loss[0]
                oc_loss = ge_loss[2]
                dot_loss = ge_loss[3]
                d_l = ge_loss[-1]

            progress_bar.update(i, values=[('total',g_loss),('g_loss',gen_loss),('dot',dot_loss),('oc',oc_loss),('style1_loss',gd),('style4_loss',d_l)])

        generator.save_weights(os.path.join('./Model',args.model_name+'_OC.hdf5'))
        if args.use_gan:        
            #generator_model.save_weights(os.path.join('./Model',args.model_name+'_OC_WGAN_G.hdf5'))
            discriminator_model.save_weights(os.path.join('./Model',args.model_name+'_OC_WGAN_D.hdf5'))
            #best_g = temp_g


def sobel_score(real,recon,fangxiang,localization=False):
    real_img = copy.deepcopy(real)
    real_blur = cv2.GaussianBlur(real_img, (0, 0), 1)
    if fangxiang==0:
        real_sobel = cv2.Sobel(real_blur,cv2.CV_16S,0,2)
        recon_sobel = cv2.Sobel(recon,cv2.CV_16S,0,2)
    else :
        real_sobel = cv2.Sobel(real_blur,cv2.CV_16S,2,0)
        recon_sobel = cv2.Sobel(recon,cv2.CV_16S,2,0)
    real_sobel = cv2.convertScaleAbs(real_sobel)
    recon_sobel = cv2.convertScaleAbs(recon_sobel)
    
    real_sobel = cv2.cvtColor(real_sobel,cv2.COLOR_BGR2GRAY)
    recon_sobel = cv2.cvtColor(recon_sobel,cv2.COLOR_BGR2GRAY)
    diff = 0
    gray_img = cv2.cvtColor(real_img,cv2.COLOR_BGR2GRAY)
    for i in range(1,len(real_sobel)-1):
        for j in range(1,len(real_sobel[0])-1):

            temp = abs(int(real_sobel[i][j])-int(recon_sobel[i][j]))
            
            if temp>20 and temp<50:
                diff+= 1
                if localization is True:
                    real_img[i][j] = [255,0,0]

    if localization is True:
        return diff,real_img
    else:
        return diff
def lmse(real,recon):
    #sns.set()
    real = cv2.GaussianBlur(real, (0, 0), 1)
    recon = cv2.GaussianBlur(recon, (0, 0), 1)
    noise = real.astype('float32')-recon.astype('float32')
    noise = np.sum(np.square(noise))



    return noise




def chayi(real,recon,boundary):

    noise = cv2.convertScaleAbs(real-recon)
    #noise = cv2.cvtColor(noise, cv2.COLOR_BGR2GRAY)
    noise[noise<=boundary]=0
    #noise = noise-int(boundary)
    
    
    #noise[noise>boundary]
    
    noise[noise<=boundary]=0
    noise = noise.astype('float32')
    noise = np.power(noise,4)/np.power(boundary,4)*4
    #noise = noise/np.max(noise)*255
    #print(np.max(noise))
    noise[noise>100]=255
    #noise = noise.astype('int32')
    #noise = noise.astype('uint8')
    noise = np.uint8(noise)
    #print(np.max(noise))
    #noise = cv2.equalizeHist(noise)
    noise = cv2.applyColorMap(noise, cv2.COLORMAP_HOT)
    #noise = cv2.GaussianBlur(noise, (0, 0), 1)


    return noise


def xception_score(src,tar):
    src_img = src.astype('float32')
    src_img = src_img/255
    src_img = src_img.reshape((1,256,256,3))
    tar_img = tar.astype('float32')
    tar_img = tar_img/255
    tar_img = tar_img.reshape((1,256,256,3))

    result_src = xception_model.predict(src_img)
    result_tar = xception_model.predict(tar_img)

    result_tar = np.array(result_tar)
    result_src = np.array(result_src)

    result =  np.sqrt(np.sum(np.square(result_tar-result_src)))




    return result


def fft_score(src,tar):
    src_f = np.fft.fft2(src)
    src_fshift = np.fft.fftshift(src_f)
    src_s = np.abs(src_fshift)

    tar_f = np.fft.fft2(tar)
    tar_fshift = np.fft.fftshift(tar_f)
    tar_s = np.abs(tar_fshift)
    result =  np.sqrt(np.sum(np.square(src_s-tar_s)))
    return result


def cos_loss(feature1,feature2):
    dot = np.sum(np.multiply(feature1, feature2),axis=1)
    norm = np.linalg.norm(feature1, axis=1) * np.linalg.norm(feature2, axis=0)
    dist = dot / norm
    return 1-dist
def ssim_sc(src, recon):
    src = cv2.GaussianBlur(src, (0, 0), 1)
    recon = cv2.GaussianBlur(recon, (0, 0), 1)

    score =  compare_ssim(src,recon, multichannel=True)
    return score


def test(save_path,X_path,length,person,style_vec,min_ssim,max_sobel,max_xception):


    X_test = train_same_people_DataSequence(path=X_path,batch_size=args.batch_size,number=person,famous=args.famous)

    #content_model = tf.keras.models.Model(inputs=generator.input,outputs=generator.get_layer("lambda").output)

    mse_results = list()
    style_results = list()
    sum_score = list()
    boundary = math.sqrt(max_sobel/(256*256*3))
    test_style_vec = list()

    for i in range(args.batch_size):
        test_style_vec.append(style_vec)

    test_style_vec = np.array(test_style_vec)
    for idx in tqdm(range(int(len(X_test)/length))):
        batch_x = X_test[idx][0][1]

        
        # results_x = generator_model.predict_on_batch(batch_x)
        # results_x[0] = np.array(results_x[0])
        # results_x[1] = np.array(results_x[1])

        result_content = content_model.predict_on_batch(batch_x)
        

        result_style = style_model.predict_on_batch(batch_x)

        

        results_x,_,_,_,_,_,_  = re_model.predict_on_batch([result_content,test_style_vec])
        re_content = content_model.predict_on_batch(results_x)
        re_style = style_model.predict_on_batch(results_x)
        re_mean_loss = result_style-re_style
        re_content_loss = result_content-re_content



        for i in range(len(results_x)):
            batch_x[i] = (np.array(batch_x[i])+1)*127.5

            results_x[i] = (results_x[i]+1)*127.5
            batch_x[i][batch_x[i]<0]=0
            batch_x[i][batch_x[i]>255]=255
            results_x[i][results_x[i]<0]=0
            results_x[i][results_x[i]>255]=255
            # ssima = cv2.GaussianBlur(batch_x[i], (0, 0), 1)
            # ssimb = cv2.GaussianBlur(results_x[i], (0, 0), 1)
            ssim_score = ssim_sc(batch_x[i],results_x[i])

            if ssim_score<0.1:
                ssim_score = 0
            elif ssim_score<min_ssim:
                ssim_score = (ssim_score-0.1)/(min_ssim-0.1)*0.5
            else:
                ssim_score = (ssim_score-min_ssim)/(1-min_ssim)*0.5+0.5

            img_test = batch_x[i]

            
            score_sobel = lmse(batch_x[i],results_x[i])
            mse_results.append(np.sqrt(score_sobel//(256*256*3)))
            #score_sobel = 1-0.5*score_sobel/(max_sobel)
            score_lmse = 1-0.5*score_sobel/(max_sobel)

            if score_lmse<0:
                score_lmse=0

                
            # vec = results_x[1][i]
            # dis_score = np.sqrt(np.sum(np.square(vec-centor))) 
            # dis_score = 1-0.5*dis_score/(max_dis)



            #score_xception = xception_score(img_test,results_x[i])
            score_xception = np.sum(abs(re_content_loss[i]))
            score_xception = 1-0.5*score_xception/(max_xception)
            if score_xception<0:
                score_xception = 0

            total_score = ssim_score*score_lmse*score_xception

            #dot_result = np.dot(results_y[0][i],results_y[1][i])

            sum_score.append([idx,i,total_score,ssim_score,score_lmse,score_xception])
        
            ch = chayi(batch_x[i],results_x[i],boundary)
            #htich = np.hstack((batch_x[i],results_x[i],ch))
            #htich = cv2.cvtColor(htich,cv2.COLOR_RGB2BGR)
            output_batch = cv2.cvtColor(batch_x[i],cv2.COLOR_RGB2BGR)
            output_result = cv2.cvtColor(results_x[i],cv2.COLOR_RGB2BGR)
            #ch = cv2.cvtColor(ch,cv2.COLOR_RGB2BGR)

            cv2.imwrite(os.path.join(save_path,str(idx)+'_'+str(i)+'source.png'),output_batch)
            cv2.imwrite(os.path.join(save_path,str(idx)+'_'+str(i)+'result.png'),output_result)
            cv2.imwrite(os.path.join(save_path,str(idx)+'_'+str(i)+'mse.png'),ch)

        #content_results.extend(results_y[1])
        #style_results.extend(results_y[0])
    return sum_score,np.array(mse_results)


def cal_score():
    X_train = train_same_people_DataSequence(path=args.train_dir,batch_size=BATCH_SIZE,number=args.person,famous=args.famous)
    centor = np.zeros((style_len,), dtype=np.float)
    style_vec = np.zeros((int(style_len*2),), dtype=np.float)
    #re_model = tf.keras.models.Model(inputs=generator.input,outputs=generator.get_layer("lambda").output)

    min_ssim = 1
    max_sobel = 0
    conter_list = list()
    ssim_list = list()
    dis_list = list()
    sobel_list = list()
    xception_list = list()


    mean_list = list()
    var_list = list()

    test_len = len(X_train)

    test_len = int(test_len)


    for idx in tqdm(range(test_len)):
        batch_x = X_train[idx][0][1]
        results_style = style_model.predict_on_batch(batch_x)
        for j in range(len(results_style)):
            style_vec+=results_style[j]
            mean_list.append(results_style[j])

    style_vec = style_vec/(test_len*args.batch_size)
    test_style_vec = list()

    for i in range(args.batch_size):
        test_style_vec.append(style_vec)

    for idx in tqdm(range(test_len)):
        batch_x = X_train[idx][0][1]
        result_content = content_model.predict_on_batch(batch_x)
        result_style = style_model.predict_on_batch(batch_x)
        test_style_vec = np.array(test_style_vec)
        results_x,_,_,_,_,_,_ = re_model.predict_on_batch([result_content,results_style])

        #
        re_content = content_model.predict_on_batch(results_x)
        re_style = style_model.predict_on_batch(results_x)
        re_style_loss = re_style-result_style
        re_content_loss = result_content-re_content
        #print(np.sum(np.square(re_mean)))
        results_x = np.array(results_x)

        # for vec in results_x[1]:
        #     centor+=vec
        #     conter_list.append(vec)

        for j in range(len(results_x)):
            batch_x[j] = (np.array(batch_x[j])+1)*127.5

            results_x[j] = (results_x[j]+1)*127.5

            batch_x[j][batch_x[j]<0]=0
            batch_x[j][batch_x[j]>255]=255
            results_x[j][results_x[j]<0]=0
            results_x[j][results_x[j]>255]=255
            #ssim_pic = compare_ssim(batch_x[j],results_x[j], multichannel=True)
            ssim_pic = ssim_sc(batch_x[i],results_x[i])
            ssim_list.append(ssim_pic)

            #x_score = xception_score(batch_x[j],results_x[j])
            x_score = np.sum(abs(re_content_loss[j]))

            #lmse_loss = np.sum(abs(re_content_loss[j]))
            lmse_loss = lmse(batch_x[j],results_x[j])
            #x_score = np.sum(abs(re_mean_loss[j]))
            #lmse_loss = xception_score(batch_x[j],results_x[j])
            sobel_list.append(lmse_loss)    
            xception_list.append(x_score)
    #centor = centor/(test_len*8)
    #max_dis = 0
    #conter_list = np.array(conter_list)
    # for i in range(len(conter_list)):
    #     dis = np.sqrt(np.sum(np.square(centor-conter_list[i])))
    #     dis_list.append(dis)


    ssim_list.sort(reverse = True)
    sobel_list.sort(reverse = False)
    #dis_list.sort(reverse = False)
    xception_list.sort(reverse = False)

    acc_lenth = int(test_len*args.batch_size*0.99)

    ssim_acc_lenth = int(test_len*args.batch_size*0.99)
    ssim_list = ssim_list[0:ssim_acc_lenth]
    sobel_list = sobel_list[0:acc_lenth]
    xception_list = xception_list[0:acc_lenth]


    min_ssim = ssim_list[-1]
    max_sobel = sobel_list[-1]
    max_xception = xception_list[-1]
    return min_ssim,max_sobel,style_vec,max_xception    


def tsne_pic(train_list,real_list,fake_list,save_name):


    train_list = np.array(train_list)
    real_list = np.array(real_list)
    fake_list = np.array(fake_list)
    c = ['r.','g.','b.','c.','m.','y.','k.']
    label=list()
    feature = np.vstack([train_list,real_list])
    feature = np.vstack([feature,fake_list])
    print (feature.shape)

    for i in range(len(train_list)):
        label.append(c[1])
    for i in range(len(real_list)):
        label.append(c[2]) 
    for i in range(len(fake_list)):
        label.append(c[0])
    tsne = TSNE(n_components=2, init='pca')
    res = tsne.fit_transform(feature)
    for i in range(len(label)):
        plt.plot(res[i][0],res[i][1],label[i])
    plt.savefig(save_name)
    plt.show()

def get_normal_pic(pic):
    pic = (pic+1)*127.5
    pic[pic<0]=0
    pic[pic>255]=255
    return pic

def get_ex_pic(X,style_vec,test_list,boundary,ismax):
    if ismax:
        show1_loc = np.argmax(test_list)
        show1 = X[int(show1_loc/args.batch_size)][0][1]
        result_content  = content_model.predict_on_batch(show1)
        predict_1,_,_,_,_,_,_ = re_model.predict_on_batch([result_content,style_vec])
        res_show = get_normal_pic(show1[int(show1_loc%args.batch_size)])
        res_predict = get_normal_pic(predict_1[int(show1_loc%args.batch_size)])
        bound = math.sqrt(boundary/(256*256*3))
        difference = chayi(res_show,res_predict,bound)
        res = np.hstack((res_show,res_predict,difference))
        
    else:    
        show1_loc = np.argmin(test_list)
        show1 = X[int(show1_loc/args.batch_size)][0][1]
        result_content  = content_model.predict_on_batch(show1)
        predict_1,_,_,_,_,_,_ = re_model.predict_on_batch([result_content,style_vec])
        res = np.hstack((get_normal_pic(show1[int(show1_loc%args.batch_size)]),get_normal_pic(predict_1[int(show1_loc%args.batch_size)])))
    return res

def test_to_show(X,boundary,style):


    #model.load_weights(os.path.join('./Model',args.model_name+'.hdf5'))
    test_list = list()


    test_style_vec = list()
    for i in range(args.batch_size):
        test_style_vec.append(style)
        
    test_style_vec = np.array(test_style_vec)
    for i in tqdm(range(int(len(X)/10))):
        batch_X = X[i][0][1]
        result_content = content_model.predict_on_batch(batch_X)
        results_X,_,_,_,_,_,_ = re_model.predict_on_batch([result_content,test_style_vec])


        for j in range(args.batch_size):
            batch_normal = get_normal_pic(batch_X[j])
            result_normal = get_normal_pic(results_X[j])
            rmse = lmse(batch_normal,result_normal)
            test_list.append(rmse)

    test_list = np.array(test_list)
    print(np.max(test_list))
    count = np.sum(test_list>=boundary)
    score = count/(len(X)*args.batch_size/10)

    if score<0.5:
        print('test video is normal')
        showed_1 = get_ex_pic(X,test_style_vec,test_list,boundary,False)
        test_list[np.argmin(test_list)] = np.max(test_list)
        showed_2 = get_ex_pic(X,test_style_vec,test_list,boundary,False)
        result = np.vstack((showed_1,showed_2))
        result = cv2.cvtColor(result,cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(args.save_path,args.person+'.png'),result)

    else:
        print('test video is abnormal')
        showed_1 = get_ex_pic(X,test_style_vec,test_list,boundary,True)
        test_list[np.argmin(test_list)] = np.max(test_list)
        showed_2 = get_ex_pic(X,test_style_vec,test_list,boundary,True)
        result = np.vstack((showed_1,showed_2))
        result = cv2.cvtColor(result,cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(args.save_path,args.person+'.png'),result)


def get_density(real,fake,faceswap,faceswap_gan):
    sns.set(color_codes=True)
 
    #ax_train = sns.kdeplot(data=train,color='b',legend=True)
    ax_real = sns.kdeplot(data=real,color='g',legend=True)
    ax_fake = sns.kdeplot(data=fake,color='r',legend=True)
    ax_fake = sns.kdeplot(data=faceswap,color='b',legend=True)
    ax_fake = sns.kdeplot(data=faceswap_gan,color='y',legend=True)
    #ax_fake = sns.kdeplot(data=lip,color='brown',legend=True)
    plt.yticks(())
    plt.legend(['Real','DeepFaceLab','FaceSwap','FaceSwap-GAN','lip sync'])
    plt.xlabel("Pixel difference")
    plt.ylim(0,1)
    ax_real.axvline(x=11 , color='black',linestyle='--')
    plt.savefig(os.path.join(args.save_path,args.person,'distribution.png'))

if args.opr == 'train': 
    # content_model.summary()
    # re_model.summary()
    train()
else:
    generator.load_weights(os.path.join('./Model',args.model_name+'_OC.hdf5'))
    if not os.path.exists(os.path.join(args.save_path,args.person,'train')):
        os.makedirs(os.path.join(args.save_path,args.person,'train'))
    if not os.path.exists(os.path.join(args.save_path,args.person,'test')):
        os.makedirs(os.path.join(args.save_path,args.person,'test'))    
    if not os.path.exists(os.path.join(args.save_path,args.person,'fake')):
        os.makedirs(os.path.join(args.save_path,args.person,'fake'))    
    if not os.path.exists(os.path.join(args.save_path,args.person,'faceswap')):
        os.makedirs(os.path.join(args.save_path,args.person,'faceswap'))
    if not os.path.exists(os.path.join(args.save_path,args.person,'faceswap_gan')):
        os.makedirs(os.path.join(args.save_path,args.person,'faceswap_gan'))        
    if not os.path.exists(os.path.join(args.save_path,args.person,'lip')):
        os.makedirs(os.path.join(args.save_path,args.person,'lip'))     
    
    min_ssim,max_sobel,centor,max_xception = cal_score()
    #print(min_ssim,max_sobel,max_xception)
    #np.savetxt(os.path.join(args.save_path,args.person,'centor.csv'), centor, delimiter=',')


    # If you want to show a demo, please use these codes.
    # X_real = train_same_people_DataSequence(path=args.real_dir,batch_size=BATCH_SIZE,number=args.person,famous=args.famous)
    # X_fake = train_same_people_DataSequence(path=args.fake_dir,batch_size=BATCH_SIZE,number=args.person,famous=args.famous)
    # test_to_show(X_real,max_sobel,centor)
    # test_to_show(X_fake,max_sobel,centor)



    test_score,real_results = test(os.path.join(args.save_path,args.person,'test'),args.real_dir,2,args.person,centor,min_ssim,max_sobel,max_xception) 
    fake_score,fake_results = test(os.path.join(args.save_path,args.person,'fake'),args.fake_dir,2,'_'+args.person+'_',centor,min_ssim,max_sobel,max_xception)

    np.savetxt(os.path.join(args.save_path,args.person,'fake_score.csv'), fake_score, delimiter=',')
    np.savetxt(os.path.join(args.save_path,args.person,'real_score.csv'), test_score, delimiter=',') 



    #tsne_pic(train_content,test_content,fake_content,os.path.join(args.save_path,args.person,'content_WGAN.png'))
    #tsne_pic(train_style,test_style,fake_style,os.path.join(args.save_path,args.person,'style_WGAN.png'))
