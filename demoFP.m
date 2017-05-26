%% Pose Optimization Demo - Full Perspective case
% Full Perspective case of our Pose Optimization.
% For convenience, we assume that the heatmaps (keypoint localizations) 
% are precomputed and provided in the folder demo.

clear
startup

datapath = 'demo/gascan/';
dict = load(sprintf('%s/annot/dict.mat',datapath));
cad = load(sprintf('%s/annot/cad.mat',datapath));
annotfile = sprintf('%s/annot/valid.mat',datapath);
load(annotfile);

for ID = 1:length(annot.imgname)

    % input
    imgname = annot.imgname{ID};
    center = annot.center(ID,:);
    scale = annot.scale(ID);
    K = annot.K{ID};

    % read heatmaps and detect maximum responses
    heatmap = h5read(sprintf('%s/exp/valid_%d.h5',datapath,ID),'/heatmaps');
    heatmap = permute(heatmap,[2,1,3]);
    [W_hp,score] = findWmax(heatmap);
    W_im = transformHG(W_hp,center,scale,size(heatmap(:,:,1)),true);
    W_ho = K\[W_im;ones(1,size(W_im,2))];
    
    % pose optimization - weak perspective
    output_wp = PoseFromKpts_WP(W_hp,dict,'weight',score,'verb',false);
    
    % pose optimization - full perspective
    output_fp = PoseFromKpts_FP(W_ho,dict,'R0',output_wp.R,'weight',score,'verb',false);
    % update translation given object metric size
    S_fp = bsxfun(@plus,output_fp.R*output_fp.S,output_fp.T);
    [model_fp,w,~,T_fp] = fullShape(S_fp,cad);
    output_fp.T_metric = T_fp/w;    
    
    % visualization
    img = imread(sprintf('%s/images/%s',datapath,imgname));
    vis_fp(img,output_fp,output_wp,heatmap,center,scale,K,cad);
    pause
    close all
    
end
