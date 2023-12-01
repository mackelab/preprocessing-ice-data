clear all
close all

% load bedmachine coordinates
[x1,y1,names] = antbounds_data('shelves','Antarctica');
[xi,yi]       = antbounds_data('Ekstrom','xy'); %Ekstrom boundaries
[xgl,ygl] = antbounds_data('gl','xy'); %Ekstrom GL
[xgrid,ygrid] = psgrid('Ekstrom Ice Shelf',[200,200],10,'xy'); %Create grid of points on Ekstrom

% load IRH transect dataset
Data = load("../data/Ekstrom/Ekstroem_flowline_GPR_IRH_data.mat")
lat = transpose(Data.lat);
lon = transpose(Data.lon);
[xdata,ydata] = ll2ps(lat,lon);


xdata2 = transpose(Data.psX);
ydata2 = transpose(Data.psY);


% take only coordinates within Ekstrom ice shelf
xmax = max(xi)+1e4;
ymax = max(yi)+1e1;
xmin = min(xi)-1e4;
ymin = min(yi)-1e1;
in = xgl<xmax & xgl >xmin & ygl>ymin  & ygl <ymax;

%Grounding line
xgl2 = transpose(xgl(in));
ygl2=transpose(ygl(in));
ptarray = [xgl2,ygl2];

%IRH transect
in = xdata<xmax &xdata >xmin & ydata >(ymin+2e2) & ydata <(ymax-1e2);
save_in = in';
shelf_mask_file = "../data/Ekstrom/LayerData_flowline_mask.mat"
disp(save_in);
save(shelf_mask_file,"save_in")
xdata = xdata(in);
ydata = ydata(in);

ydatamax = max(ydata);


%find close points to the transect near the grounding line, and calculate
%flowtubes to either side of it.
[minValue,closestIndex] = min(sqrt(sum((ptarray-[-3e5, 1.98e6]).^2,2)));
[xleft,yleft] = itslive_flowline(ptarray(closestIndex,1),ptarray(closestIndex,2));
in = xleft<xmax & xleft >xmin & yleft>ymin  & yleft <ydatamax+1e3;
xleft = xleft(in);
yleft = yleft(in);



[minValue,closestIndex] = min(sqrt(sum((ptarray-[-2.98e5, 1.98e6]).^2,2)));
[xright,yright] = itslive_flowline(ptarray(closestIndex,1),ptarray(closestIndex,2));
in = xright<xmax & xright >xmin & yright>ymin  & yright <ydatamax+1e3;
xright = xright(in);
yright = yright(in);


%Potentially plot things
% hold
% plot(xi,yi);
% plot(xgrid,ygrid,'go');
% plot(xgl2,ygl2,'ro');
% plot(ptarray(closestIndex,1),ptarray(closestIndex,2),'bo');
% plot(xleft,yleft,'g--');
% plot(xright,yright,'g--');
% plot(xdata,ydata,'k-');
% plot(xdata2,ydata2,'m-')
% disp(size(xdata));
% disp(size(xleft));
% disp(size(xright));
% disp(size(nan(size(xdata,1)-size(xleft,1),1)));


%output to .csv file
xleft = [xleft; nan(size(xdata,1)-size(xleft,1),1)];
xright = [xright; nan(size(xdata,1)-size(xright,1),1)];
yleft = [yleft; nan(size(xdata,1)-size(yleft,1),1)];
yright = [yright; nan(size(xdata,1)-size(yright,1),1)];
M = [xdata,ydata,xleft,yleft,xright,yright];
T = array2table(M);
T.Properties.VariableNames = {'xdata','ydata','xleft','yleft','xright','yright'};

writetable(T,'../data/Ekstrom/flowtube_test.csv');

