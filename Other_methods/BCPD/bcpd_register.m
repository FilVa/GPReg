function [template_transf,corr_target, success] = bcpd_register(target,template,opt)


%% set 'win=1' if windows
win=1;
%% input files
%y   =sprintf('%s/../data/face-x.txt',pwd);
%x   =sprintf('%s/../data/face-y.txt',pwd);
fnm =sprintf('%s/../bcpd',           pwd);
fnw =sprintf('%s/../win/bcpd.exe',   pwd);


% INSERT HERE THE PATH TO BCPD EXECUTABLE
fnw = '\BCPD\demo/../win/bcpd.exe';



if(win==1) bcpd=fnw; else bcpd=fnm; end;

x = target;
y = template;

%% parameters

% general parameters
omg = num2str(opt.omg);
lmd = num2str(opt.lmd);
bet = num2str(opt.beta);
gma = num2str(opt.gamma);
dist = num2str(opt.dist);
K   ='150';%150
J   ='300';%300

% convergence
c   =num2str(opt.tol);
n   =num2str(opt.max_loops);
N   =num2str(opt.min_loops);

norm = opt.scale;
%% execution
prm1=sprintf('-w%s -b%s -l%s -g%s -e%s -u%s',omg,bet,lmd,gma,dist,norm); % commands for general parameters
prm2=sprintf('-c%s -n%s -N%s ',c,n,N); % convergence

if(opt.flag_acc == 0)
    disp('No acceleration')
    cmd =sprintf('%s -x%s -y%s %s %s -syce',bcpd,x,y,prm1,prm2); %-sA : to ouput all files
    disp(cmd)
else
    disp('Acceleration with standard parameters')
    cmd =sprintf('%s -x%s -y%s %s %s -syce -A',bcpd,x,y,prm1,prm2); %-sA : to ouput all files. -A for acceleration with standard parameters
end    
%cmd =sprintf('%s -x%s -y%s %s %s %s -DY,1000,0 -sA',bcpd,x,y,prm1,prm2,prm3);
%cmd =sprintf('%s -x%s -y%s %s %s %s -syxuvceaPT',bcpd,x,y,prm1,prm2,prm3);
disp(pwd)
system(cmd); 
%optpath3;
disp(pwd)
try
    % successful registration so outputfiles exist
    template_transf = readmatrix('output_y.txt');
    non_outlier_labels = readmatrix('output_c.txt');
    matched_points = readmatrix('output_e.txt');
    corr_target = matched_points(:,2);
    corr_target(non_outlier_labels(:,2)==0) = nan;
    success = 1;

catch
    disp('ERROR in BCPD')
    template_transf = nan;
    
    corr_target = nan;
    success = 0;

end



end

