#!/bin/bash

rm -rf dataset
mkdir dataset
cd dataset

CPN_2D_ID='1Q_u_xFeUFzUeJgx_Y4ODm3PlF72oD9ow'
CPN_NAME='data_2d_h36m_cpn_ft_h36m_dbb.npz'
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$CPN_2D_ID -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$CPN_2D_ID" -O $CPN_NAME && rm -rf /tmp/cookies.txt


GT_2D_ID='1PDEJV_SZPptoW7dMQ5E1pHrS-Ix9uqUZ'
GT_2D_NAME='data_2d_h36m_gt.npz'
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$GT_2D_ID -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$GT_2D_ID" -O $GT_2D_NAME && rm -rf /tmp/cookies.txt


GT_3D_ID='14s7QZ0J_C2TifdPQ1EqCNoiJogsi8y3i'
GT_3D_NAME='data_3d_h36m.npz'
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$GT_3D_ID -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$GT_3D_ID" -O $GT_3D_NAME && rm -rf /tmp/cookies.txt

cd ../