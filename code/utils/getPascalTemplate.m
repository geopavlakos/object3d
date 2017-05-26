function dict = getPascalTemplate(model)

p = length(model.pnames);

S = nan(3,p);
for j = 1:p
    xyz = model.(model(1).pnames{j});
    if ~isempty(xyz)
        S(:,j) = xyz';
    end
end

kpt_id = find(mean(~isnan(S))==1);
kpt_name = model.pnames(kpt_id);
S = S(:,kpt_id);

dict.B = normalizeS(S);
dict.mu = dict.B;
dict.pc = [];

dict.kpt_id = kpt_id;
dict.kpt_name = kpt_name;
dict.model_id = 1;