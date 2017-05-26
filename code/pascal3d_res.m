function pascal3d_res(datapath,savepath)

    annotfile = sprintf('%s/valid.mat',datapath);
    load(annotfile);

    testlist = find(~annot.occluded & ~annot.truncated);
    
    %%
    numClass = max([annot.classID{:}]);
    ERR1 = cell(numClass,1);
    className = cell(numClass,1);
    for i = 1:length(annot.class)
        className{annot.classID{i}} = annot.class{i};
    end

    %%
    for i = testlist'

        savefile = sprintf('%s/valid_%d.mat',savepath,i);
        load(savefile);

        classID = annot.classID{i};R_gt = annot.rot{i};

        R = (diag([1,-1,-1])*output_wp.R)';
        err_R = 180/pi*norm(logm(R_gt'*R),'fro')/sqrt(2);
        if isnan(err_R)
            err_R = 90;
        end
        ERR1{classID} = [ERR1{classID},err_R];
    end

    %%
    fid = fopen(sprintf('%s/../res.txt',savepath),'a+');
    fprintf(fid,'%s: %s\n',datestr(now),savepath);
    fprintf(fid,'%13s','');
    for i = 1:length(className)
        if ~isempty(className{i})
            fprintf(fid,'& %13s ',className{i});
        end
    end
    fprintf(fid,'& %13s','Average');
    fprintf(fid,'\\\\\n');

    med = [];
    for i = 1:length(ERR1)
        if ~isempty(ERR1{i})
            med = cat(1,med,median(ERR1{i}));
        end
    end

    fprintf(fid,'%13s','MedErr ');
    for i = 1:length(med)
        fprintf(fid,'& %13s ',sprintf('%.2f',med(i)));
    end
    fprintf(fid,'& %13s ',sprintf('%.2f',mean(med)));
    fprintf(fid,'\\\\\n');

    fclose(fid);
    
end