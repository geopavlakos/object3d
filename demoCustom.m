%% Demo
% This file will help you use our pipeline on your custom images.

%% 0) Training keypoint localization model (Optional)
% Since we provide a pretrained model on PASCAL3D, you can skip this step.
% However, for your convenience, we also provide the training code, to
% allow training your own model on PASCAL3D. 
% This is a modified version of the original stacked hourglass released code.
% For training, you can run:
%
% cd pose-hg/pose-hg-train/src
% th main.lua -dataset pascal3d -expID test-run-stacked -netType hg-stacked -task pose-int -nStack 2 -LR 2.5e-4 -nEpochs 100 -snapshot 1
%
% You can use the resulting models for testing instead of our pretrained version.

%% 1) Creating the dataset
% First you need to create the dataset for the stacked hourglass ConvNet
% We read from a .mat file that we have stored the following information for each image:
% a) imgname
% b) center (center of the bounding box)
% c) scale (defined based on the size of the bounding box as max(width,length)/200 )
% Additionally, for the pose optimization we will need:
% d) class
% e) cad_index (if we know the CAD model instance, otherwise set cadSpecific = 0 later)

dataset = 'pascal3d-sample';

load demo/pascal3d-sample/annot/valid.mat
datapath = sprintf('pose-hg/pose-hg-demo/data/%s/',dataset);

% copy images
imagespath = sprintf('%s/images/',datapath);
mkdir(imagespath);
copyfile('demo/pascal3d-sample/images/*', imagespath);

annotpath = sprintf('%s/annot/',datapath);
mkdir(annotpath);

% txt file with image names
N = numel(annot.imgname);
fid = fopen(sprintf('%s/valid_images.txt',annotpath),'w');
for i = 1:N
    fprintf(fid,sprintf('%s.jpg\n',annot.imgname{i}));
end
fclose(fid);

% h5 file with annotations
h5file = sprintf('%s/valid.h5',annotpath);
h5create(h5file,'/center',[2 N]);
h5create(h5file,'/scale', N);
h5write(h5file, '/center', annot.center');
h5write(h5file, '/scale', annot.scale);
save(sprintf('%s/valid.mat',annotpath),'annot');

%% 2) Running the stacked hourglass convnet

% We do this outside matlab. You need to run:
% cd pose-hg/pose-hg-demo
% th main.lua pascal3d-sample valid pretrained

%% 3) Running the pose optimization

% the code reads the output heatmaps and optimizes for the 3D pose

predpath = sprintf('%s/../../exp/%s/',datapath,dataset);
cadSpecific = 1; % if you don't know the cad index for the instance, set this variable to 0

annotfile = sprintf('%s/annot/valid.mat',datapath);
load(annotfile);

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
    heatmap = h5read(sprintf('%s/valid_%d.h5',predpath,ID),'/heatmaps');
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
