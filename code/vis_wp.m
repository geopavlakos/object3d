function vis_wp(img, opt, heatmap, center, scale, cad, dict)

    img_crop = cropImage(img,center,scale);
    S = bsxfun(@plus,opt.R*opt.S,[opt.T;0]);
    model = fullShape(S,cad,dict.kpt_id);
    mesh2d = model.vertices(:,1:2)'*200/size(heatmap,2);
    
    % visualization
    nplot = 4;
    h = figure('position',[100,100,nplot*300,300]);
    % cropped image
    subplot('position',[0 0 1/nplot 1]);
    imshow(img_crop); hold on;
    % heatmap
    subplot('position',[1/nplot 0 1/nplot 1]);
    response = sum(heatmap,3);
    max_value = max(max(response));
    mapIm = imresize(mat2im(response, jet(100), [0 max_value]),[200,200],'nearest');
    imToShow = mapIm*0.5 + (single(img_crop)/255)*0.5;
    imagesc(imToShow); axis equal off
    % project cad model on image
    subplot('position',[2/nplot 0 1/nplot 1]);
    imshow(img_crop); hold on;
    patch('vertices',mesh2d','faces',model.faces,'FaceColor','blue','FaceAlpha',0.3,'EdgeColor','none');
    % object viewpoint
    subplot('position',[3/nplot 0 1/nplot 1]);
    h2 = subplot('position',[3/nplot 0 1/nplot 1]);
    trisurf(model.faces,model.vertices(:,1),model.vertices(:,2),model.vertices(:,3),'EdgeColor','none');axis equal;
    view(0,-90);
    set(h2,'XTick',[],'YTick',[]);
    set(h2,'visible','off')

end
