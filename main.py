#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 19:59:56 2024

@author:
"""



import numpy as np
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from os import listdir
# import os.path
import model_defs
import pandas as pd
import nibabel as nib
import sys




DATA_DIR= '/home/ch209389/Downloads/Fetal_CD/'

TEST_DIR= sys.argv[1]


MODEL_PATH= DATA_DIR + 'model/model_133_8132_8651_8229_7515.ckpt'
# MODEL_PATH= '/local/Fetal_dMRI_complete_seg/training/trial_55/model/model_25_8076_8626_8142_7461.ckpt'
                        
                        

# labels_df_seg=       pd.read_csv( DATA_DIR + 'labelkey.csv' , delimiter= ',')
# label_names_seg=     labels_df_seg.loc[:,"label"].to_numpy()
# label_convr_map_seg= labels_df_seg.loc[:,["id","seg"]].to_numpy()
# label_young_seg=     labels_df_seg.loc[:,"young"].to_numpy()
# label_old_seg=       labels_df_seg.loc[:,"old"].to_numpy()

# labels_df_par= np.loadtxt( DATA_DIR + 'labelkey_parcel.csv' , delimiter= ',')


TRACT_NAMES= pd.read_csv( DATA_DIR + 'TRACT_NAMES.csv' , delimiter= ',', header=None)
TRACT_NAMES= list(TRACT_NAMES[0])


PARC_LABELS= pd.read_csv( DATA_DIR + 'labelkey_parcel.csv' , delimiter= ',', header=None)
PARC_LABELS= list(PARC_LABELS[0])

# tract_labels_df= pd.read_csv( DATA_DIR + 'tract_labels.csv' , delimiter= ',')
# tract_names_unique= np.unique( list(tract_labels_df['names_unique']) )


N_CHANNEL= 6

N_CLASS_SEG=  5
N_CLASS_PAR= 97
N_CLASS_TRK= 31



n_class= N_CLASS_SEG + N_CLASS_PAR + N_CLASS_TRK


gpu_ind= 0


LX= LY= LZ = 64



LXc= (LX-10)//2
LYc= (LY-10)//2
LZc= (LZ-10)//2

ViT_DEPTH = 4
KERN_SIZE = 3

X =     tf.placeholder("float32", [None, LX, LY, LZ, N_CHANNEL])
Y_5tt = tf.placeholder("float32", [None, LX, LY, LZ, N_CLASS_SEG])
Y_par = tf.placeholder("float32", [None, LX, LY, LZ, N_CLASS_PAR])
Y_trk = tf.placeholder("float32", [None, LX, LY, LZ, N_CLASS_TRK])
mask_vector = tf.placeholder("float32", [None, N_CLASS_TRK])


L_RATE = tf.placeholder("float")
p_keep_conv   = tf.placeholder("float")





###### Sequential - Uncertainty weighting 2 ViT


M_patch_ViT= 8
L_patch_ViT= 8
ViT_RESIDUAL=  True
ViT_use_token=  False
ViT_layer_norm= False
ViT_learn_embd= True
ViT_d_emb= 512
ViT_n_head= 2
ViT_DEPTH= 4
ViT_Weight_STD= 0.2

N_patch_ViT= M_patch_ViT**3
VIT_d_patch= L_patch_ViT**3*1
ViT_d_emb_h= ViT_d_emb//ViT_n_head

X_ViT = tf.placeholder("float32", [None, N_patch_ViT, VIT_d_patch ])
T_5tt = tf.placeholder("float32", [None, 1, ViT_d_emb ])

N_FEAT_SEG= 32
N_FEAT_PAR= 32
N_FEAT_TRK= 32

N_FEAT_ViT= 6

F_5tt_ViT = model_defs.ViT_seg_full_patch(X_ViT, T_5tt, N_patch_ViT, VIT_d_patch, ViT_d_emb, ViT_DEPTH,
                                        ViT_n_head, ViT_d_emb_h, n_class=N_FEAT_ViT, ViT_use_token=False,
                                        ViT_RESIDUAL=True, ViT_layer_norm=True, ViT_learn_embd=False, 
                                        p_keep=1.0, ViT_Weight_STD= ViT_Weight_STD)

X2 = tf.concat([X, F_5tt_ViT], 4)

N_CHANNEL_5tt= N_CHANNEL+N_FEAT_ViT

logits_5tt, features_5tt = model_defs.my_unet_return_last_f(X2, KERN_SIZE, ViT_DEPTH, N_FEAT_SEG, N_CHANNEL_5tt, N_CLASS_SEG,\
                                  p_keep_conv, bias_init=0.001)

X3 = tf.concat([X2, features_5tt], 4)

N_CHANNEL_trk= N_CHANNEL+N_FEAT_ViT+2*N_FEAT_SEG

logits_trk, features_trk = model_defs.my_unet_return_last_f(X3, KERN_SIZE, ViT_DEPTH, N_FEAT_TRK, N_CHANNEL_trk, N_CLASS_TRK,\
                                  p_keep_conv, bias_init=0.001)

X4 = tf.concat([X2, features_trk], 4)

N_CHANNEL_par= N_CHANNEL+N_FEAT_ViT+2*N_FEAT_TRK

logits_par, features_par = model_defs.my_unet_return_last_f(X4, KERN_SIZE, ViT_DEPTH, N_FEAT_PAR, N_CHANNEL_par, N_CLASS_PAR,\
                                  p_keep_conv, bias_init=0.001)

Task_W = tf.Variable(tf.constant(0.0, shape=[N_CLASS_SEG + N_CLASS_PAR + N_CLASS_TRK]))

Y_pred_5tt = tf.nn.softmax(logits_5tt)
Y_pred_par = tf.nn.softmax(logits_par)
Y_pred_trk = tf.nn.sigmoid(logits_trk)

cost_5tt= model_defs.cost_dice_weighted(Y_5tt, Y_pred_5tt,\
                                        Task_W[:N_CLASS_SEG], n_class= N_CLASS_SEG,\
                                        loss_type='sorensen', smooth = 1e-5)

cost_par= model_defs.cost_dice_weighted(Y_par, Y_pred_par,\
                                        Task_W[N_CLASS_SEG:N_CLASS_SEG+N_CLASS_PAR], n_class= N_CLASS_PAR,\
                                        loss_type='sorensen', smooth = 1e-5)
        
cost_trk= model_defs.cost_dice_selected_weighted(Y_trk, Y_pred_trk, mask_vector,\
                                                Task_W[N_CLASS_SEG+N_CLASS_PAR:], n_class= N_CLASS_TRK,\
                                                loss_type='sorensen', smooth = 1e-5)

cost_5tt+= model_defs.cost_x_entropy(Y_5tt, logits_5tt)
cost_par+= model_defs.cost_x_entropy(Y_par, logits_par)

cost_total= cost_5tt +\
            cost_par +\
            cost_trk +\
            tf.reduce_sum(Task_W)

optimizer = tf.train.AdamOptimizer(L_RATE).minimize(cost_total)






os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ind)
saver = tf.train.Saver(max_to_keep=50)

sess = tf.Session()
sess.run(tf.global_variables_initializer())




BATCH_SIZE = 1

KEEP_TEST=  1.0















#   test



saver.restore(sess, MODEL_PATH)




test_images= [f for f in listdir(TEST_DIR) if 'nii.gz' in f]
test_images.sort()


for i_test, test_image in enumerate(test_images):
    
    print('Processing image ', test_image)
    
    tensor_img= nib.load( TEST_DIR + test_image )
    
    tensor_np= tensor_img.get_fdata()
    tensor_np= np.squeeze(tensor_np)
    
    assert tensor_np.ndim==4
    SX, SY, SZ, n_channel_= tensor_np.shape
    assert n_channel_==6
    
    SLIDE_OVERLAP= max(LX//4, 8)
    lx_list= np.squeeze( np.concatenate( (np.arange(0, SX-LX, SLIDE_OVERLAP)[:,np.newaxis] ,\
                                          np.array([SX-LX])[:,np.newaxis] ) ) )
    ly_list= np.squeeze( np.concatenate( (np.arange(0, SY-LY, SLIDE_OVERLAP)[:,np.newaxis] ,\
                                          np.array([SY-LY])[:,np.newaxis] ) ) )
    lz_list= np.squeeze( np.concatenate( (np.arange(0, SZ-LZ, SLIDE_OVERLAP)[:,np.newaxis] ,\
                                          np.array([SZ-LZ])[:,np.newaxis] ) ) )
    
    y_pred_5tt = np.zeros((SX, SY, SZ, N_CLASS_SEG))
    y_pred_par = np.zeros((SX, SY, SZ, N_CLASS_PAR))
    y_pred_trk = np.zeros((SX, SY, SZ, N_CLASS_TRK))
    
    run_count = np.zeros((SX, SY, SZ))
    
    tensor_np= tensor_np[np.newaxis,:,:,:,:]
    
    for lx in lx_list:
        for ly in ly_list:
            #print(ly)
            for lz in lz_list:
                
                if np.min(tensor_np[:,lx+LXc:lx+LX-LXc,ly+LYc:ly+LY-LYc,lz+LZc:lz+LZ-LZc,0])>0:
                    
                    batch_x = tensor_np[:, lx:lx + LX, ly:ly + LY, lz:lz + LZ, :].copy()
                    
                    ###################################################################
                    batch_x_vit = np.zeros( ( BATCH_SIZE, N_patch_ViT, VIT_d_patch ) )
                    i_patch= 0
                    block = batch_x.copy()
                    for ib in range(M_patch_ViT):
                        for jb in range(M_patch_ViT):
                            for kb in range(M_patch_ViT):
                                patch= block[0,  L_patch_ViT*ib:L_patch_ViT*(ib+1), L_patch_ViT*jb:L_patch_ViT*(jb+1),\
                                             L_patch_ViT*kb:L_patch_ViT*(kb+1), :1 ]
                                batch_x_vit[0, i_patch, :]= patch.flatten()
                    ###################################################################
                    
                    y_pred_5tt_patch, y_pred_par_patch, y_pred_trk_patch=\
                                            sess.run((Y_pred_5tt,Y_pred_par, Y_pred_trk),\
                                                      feed_dict={X: batch_x, X_ViT: batch_x_vit,\
                                                                 p_keep_conv: KEEP_TEST})
                                                
                    y_pred_5tt[lx:lx + LX, ly:ly + LY, lz:lz + LZ, :] += y_pred_5tt_patch[0, :, :, :, :]
                    y_pred_par[lx:lx + LX, ly:ly + LY, lz:lz + LZ, :] += y_pred_par_patch[0, :, :, :, :]
                    y_pred_trk[lx:lx + LX, ly:ly + LY, lz:lz + LZ, :] += y_pred_trk_patch[0, :, :, :, :]
                    
                    run_count[lx:lx + LX, ly:ly + LY, lz:lz + LZ] += 1
    
    
    y_pred_5tt = np.argmax(y_pred_5tt, axis=-1)
    y_pred_5tt[run_count == 0] = 0
    y_pred_par = np.argmax(y_pred_par, axis=-1)
    y_pred_par[run_count == 0] = 0
    
    for i_class in range(N_CLASS_TRK):
        temp= y_pred_trk[:,:,:,i_class]/(run_count+1e-9)
        temp[run_count==0]= 0
        y_pred_trk[:,:,:,i_class]= temp.copy()
    
    
    # y_pred_par_new= np.zeros(np.append(y_pred_par.shape, len(PARC_LABELS)), np.float32)
    y_pred_par_new= np.zeros(y_pred_par.shape, np.float32)
    for i_class in range(len(PARC_LABELS)):
        mask_temp= y_pred_par==i_class
        # y_pred_par_new[mask_temp,i_class]= PARC_LABELS[i_class]
        y_pred_par_new[mask_temp]= PARC_LABELS[i_class]
    # y_pred_par_new = np.argmax(y_pred_par_new, axis=-1)
    # y_pred_par_new[run_count == 0] = 0
    
    seg_2_save=  nib.Nifti1Image(y_pred_par.astype(np.float32), tensor_img.affine)
    nib.save(seg_2_save, TEST_DIR + test_image.replace('.nii','_par.nii') )
    
    seg_2_save=  nib.Nifti1Image(y_pred_par_new.astype(np.float32), tensor_img.affine)
    nib.save(seg_2_save, TEST_DIR + test_image.replace('.nii','_par_new.nii') )
    
    seg_2_save=  nib.Nifti1Image(y_pred_5tt.astype(np.float32), tensor_img.affine)
    nib.save(seg_2_save, TEST_DIR + test_image.replace('.nii','_seg.nii') )
    
    for i_trk in range(N_CLASS_TRK):
        temp= y_pred_trk[:,:,:,i_trk]>0.5
        seg_2_save=  nib.Nifti1Image(temp.astype(np.float32), tensor_img.affine)
        nib.save(seg_2_save, TEST_DIR + test_image.replace('.nii', '_' + TRACT_NAMES[i_trk] + '.nii' ))
        




























