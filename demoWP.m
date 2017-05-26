%% Pose Optimization Demo - Weak Perspective case
% Weak Perspective case of our Pose Optimization.
% For convenience, we assume that the heatmaps (keypoint localizations) 
% are precomputed and provided in the folder demo.

clear
startup

datapath = 'demo/pascal3d-sample/';
annotfile = sprintf('%s/annot/valid.mat',datapath);
load(annotfile);

% determines shape model used for pose optimization
% if set to 1 then we use the particular cad model instance
% if set to 0 then we use the deformable shape model
cadSpecific = 1;

for ID = 1:length(annot.imgname)
    
    % input
    imgname = annot.imgname{ID};
    center = annot.center(ID,:);
    scale = annot.scale(ID);
    class = annot.class{ID};
    indices = annot.indices{ID};
    cadID = annot.cad_index(ID);
    
    cad = load(sprintf('cad/%s.mat',class));
    cad = cad.(class);
    cad = cad(cadID);
    
    if cadSpecific
        dict = getPascalTemplate(cad);
    else
        dict = load(sprintf('dict/pca-%s.mat',class));
    end
    % read heatmaps and detect maximum responses
    heatmap = h5read(sprintf('%s/exp/valid_%d.h5',datapath,ID),'/heatmaps');
    heatmap = permute(heatmap(:,:,indices(dict.kpt_id)),[2,1,3]);
    [W_hp,score] = findWmax(heatmap);
    
    % pose optimization - weak perspective
    output_wp = PoseFromKpts_WP(W_hp,dict,'weight',score,'verb',false,'lam',1);
    
    % visualization
    img = imread(sprintf('%s/images/%s.jpg',datapath,imgname));
    vis_wp(img,output_wp,heatmap,center,scale,cad,dict);
    pause
    close all

end
