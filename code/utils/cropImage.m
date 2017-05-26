function im1 = cropImage(im,center,scale)

w = 200*scale;
h = w;
x = center(1) - w/2;
y = center(2) - h/2;
bbox = [x,y,w,h];
padsize = round(bbox(3:4));
im = padarray(im,padsize,0);
im1 = imcrop(im,[bbox(1:2)+padsize,bbox(3:4)]);
im1 = imresize(im1,[200,200]);
