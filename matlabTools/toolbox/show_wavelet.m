function show_wavelet(wp,x,y)
msize = 100;
[Fs,s_ind]=sort(abs(wp),'descend');
scatter(x(s_ind),y(s_ind),msize,wp(s_ind),'.');
caxis([-max(wp) max(wp)]);
hcb=colorbar('location','east');
set(gca,'Xtick',[]);
% set(gca,'Ytick',[]);
% cxt=get(hcb,'Ytick');
% cxt=[cxt(1),0,cxt(end)];
% set(hcb,'Ytick',cxt);
cpos=get(hcb,'Position');
cpos(3)=.02; % make colorbar thinner
set(hcb,'Position',cpos);
axis equal
axis off
end