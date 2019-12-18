%% generate star and planet barycentric positions
npoints = 30*15; %15 sec animation at 30 fps.

%body sizes
R = 0.075;
Rs = 0.1;

%orbital parameters
a = 1;
e = 0.1;

%gravitational parameters
mus = 2.959122082855911e-04*365.2564^2;
mup = mus/25;
mu = mus+mup;
%mu = n^2/a^3;
%mup = mu - mus;
n = sqrt(mu/a^3); 

P = 2*pi*n;
t = linspace(0,P,npoints);
M = 2*pi/P*t;

omega = 10*pi/180; %70
I = 87*pi/180;
Omega = 30*pi/180; %10

%newton-raphson to figure out E:
counter = 0;
del = 1;
E = M;
while ((del > 1e-8) && (counter <1000))
    E = E - (M - E + e.*sin(E))./(e.*cos(E)-1);
    del = sum(abs(M - (E - e.*sin(E))));
    counter = counter+1;
end
%calculate planetary distance, semi-minor axis and velocity
r = a.*(1-e.*cos(E));

A = ([a.*(cos(Omega).*cos(omega) - sin(Omega).*cos(I).*sin(omega)),...
     a.*(sin(Omega).*cos(omega) + cos(Omega).*cos(I).*sin(omega)),...
     a.*sin(I).*sin(omega)]).';

B = ([-a.*sqrt(1-e.^2).*(cos(Omega).*sin(omega) + sin(Omega).*cos(I).*cos(omega)),...
      a.*sqrt(1-e.^2).*(-sin(Omega).*sin(omega) + cos(Omega).*cos(I).*cos(omega)),...
      a.*sqrt(1-e.^2).*sin(I).*cos(omega)]).';

  
Edot = n./(1 - e*cos(E));
Eddot = -e*n^2*sin(E)./(1 - e*cos(E)).^3;

r_ps = A*(cos(E) - e) + B*(sin(E));
v_ps = A*(-sin(E).*Edot) + B*(cos(E).*Edot);
a_ps = A*(-sin(E).*Eddot - cos(E).*Edot.^2) + B*(cos(E).*Eddot - sin(E).*Edot.^2);

%barycentric positions
r_pb = mus*r_ps/mu;
r_sb = -mup*r_ps/mu;
v_pb = mus*v_ps/mu;
v_sb = -mup*v_ps/mu;
a_pb = mus*a_ps/mu;
a_sb = -mup*a_ps/mu;

%photometry
s = sqrt(r_ps(1,:).^2 + r_ps(2,:).^2); %projected separation
beta = acos(-r_ps(3,:)./r); %phase angle
Phi = (sin(beta)+(pi - beta).*cos(beta))/pi; %Lambert Phase
Fp = Phi./r.^2;
Fp = Fp/max(Fp);

Fs = ones(size(t));
inds = (Rs*abs(1 - R/Rs) < s) & (Rs+R >= s);
k0 = acos((R^2 + s(inds).^2 - Rs^2)/2/R./s(inds));
k1 = acos((Rs^2 + s(inds).^2 - R^2)/2/Rs./s(inds));
Fs(inds) = 1 - ((R/Rs)^2*k0 + k1 - sqrt( (s(inds)/Rs).^2 - (Rs^2 + s(inds).^2 - R^2).^2/4/Rs^4 ))/pi;
Fs(s <= Rs - R) = 1 - (R/Rs)^2;

%account for secondary eclipse
inds = (Fs < 1) & (r_ps(3,:) < 0);
Fs(inds) = 1 - (1 - Fs(inds))/10;

%% set up plots

[cx,cy,cz] = sphere(100);
th = 0:pi/100:2*pi;
x = cos(th);
y = sin(th);

mnmx = [min(r_pb(:))-R,max(r_pb(:))+R];
ax = repmat(mnmx,1,3);

cmap = colormap('Jet');
cmap = cmap(8:56,:);
ncolr = length(cmap);

rvs = -v_sb(3,:);
rvs = (rvs - min(rvs))/(max(rvs) - min(rvs)) * (ncolr - 1) + 1;

f = figure(576);
set(f,'Position',[97 ,63, 1600, 900],'Color','w');
%set(f,'Position',[0,0, 800, 450],'Color','w');
clf
cla reset

%animation
s1 = subplot(2,2,[1,3]);
set(s1,'Position',[0.01,0.01,0.49,0.99])
l1 = light('Position',[0 0 0],'Style','local');
sun = surface(cx*Rs+r_sb(1,1),cy*Rs+r_sb(2,1),cz*Rs+r_sb(3,1),'FaceLighting','none','FaceColor',interp1(1:ncolr,cmap,rvs(1)),'LineStyle','none');

set(gca,'Color','k','XTick',[],'YTick',[],'ZTick',[],'FontSize',16)
axis equal
hold on
axis(ax)

%the planet:
props.AmbientStrength = 0;  %no ambient light
props.DiffuseStrength = 1;  %full diffuse illumination
props.SpecularColorReflectance = 0.5;
props.SpecularExponent = 1;
props.SpecularStrength = 0.45;
props.FaceColor= [0,0,1];
props.EdgeColor = 'none';
props.FaceLighting = 'gouraud';
planet = surface(cx*R+r_pb(1,1),cy*R+r_pb(2,1),cz*R+r_pb(3,1),props);
view(2)

%plot 1
s2 = subplot(2,2,2);
set(s2,'Position',[0.585,0.58,0.4,0.4],'FontName','Georgia','FontSize',16,'XTickLabel',[])
hold on
pflux = plot(t(1)/max(t),Fp(1),'b','Linewidth',3);
xlim([0,1])
ylim([0,1])
ylabel('Normalized Planet Flux')
% p0 = get(get(s2,'YLabel'),'Position');
% set(get(s2,'YLabel'),'Position',p0+[0.015,0,0])

%plot 2
s3 = subplot(2,2,4);
set(s3,'Position',[0.585,0.125,0.4,0.4],'FontName','Georgia','FontSize',16)
xlabel('Fraction of Period')
ylabel('Normalized Total Flux')
hold on
tflux = plot(t(1)/max(t),Fs(1),'r','Linewidth',3);
xlim([0,1])
ylim([0.5,1.05])
%%
doWrite = true;
fname = 'methods0';
%%
figure(f)
if doWrite
    set(gca,'nextplot','replacechildren');
    set(f,'Visible','off','Renderer','zbuffer')
    %vidObj = VideoWriter(fname,'Uncompressed AVI');
    vidObj = VideoWriter(fname,'MPEG-4');
    vidObj.Quality = 100;
    open(vidObj);
end

%if doWrite, mov = avifile(fname); end
for i=1:length(r_pb)
    disp(i/length(r_pb))
    set(planet,'XData',cx*R+r_pb(1,i),'YData',cy*R+r_pb(2,i),'ZData',cz*R+r_pb(3,i))
    set(sun,'XData',cx*Rs+r_sb(1,i),'YData',cy*Rs+r_sb(2,i),'ZData',cz*Rs+r_sb(3,i),'FaceColor',interp1(1:ncolr,cmap,rvs(i)))
    set(pflux,'XData',t(1:i)/max(t),'YData',Fp(1:i));
    set(tflux,'XData',t(1:i)/max(t),'YData',Fs(1:i));
    
    if doWrite 
       %mov = addframe(mov,f);
       writeVideo(vidObj,getframe(f));
    else
        pause(1/30)
    end
end
if doWrite
    %mov = close(mov); 
    close(vidObj);
    set(f,'Visible','on')
end

