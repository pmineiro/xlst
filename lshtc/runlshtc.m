% quick roadmap:
%
% the routine xhat() corresponds to the eigenvalue problem associated
% with learning a tree node

function res=runlshtc(varargin)
  [mypath,~,~]=fileparts(mfilename('fullpath'));
  addpath(fullfile(mypath,'..','matlab'));

  % change this if you want, e.g., to an SSD drive location
  mmapprefix=cd;

  randn('seed',90210);
  rand('seed',8675309);

  if (nargin == 0 || varargin{1})
    rawfile='lshtcmanik.mat';
    getfrom='http://1drv.ms/1YiHLmv';
    fprintf('using Manik Varma train-test split\n');
  else
    rawfile='lshtcmunge.mat';
    getfrom='http://1drv.ms/1MprMOn';
    fprintf('using Paul Mineiro train-test split\n');
  end

  start=tic;
  fprintf('loading data ... ');

  if (exist(fullfile(cd,rawfile),'file') == 2)
    load(rawfile);
  else
    error('you must download %s available from %s\n',rawfile,getfrom);
  end
  toc(start)

  [~,n]=size(xttic);
  [~,m]=size(xstic);
  [c,~]=size(yttic);

  subn=ceil(0.1*n);
  subtrain=sort(randperm(n,subn));
  subxttic=xttic(:,subtrain);
  subyttic=yttic(:,subtrain);

  start=tic;
  res=endtoendtree(xttic,yttic,xstic,ystic,...
                   struct('depth',14,'s',4000,'smin',3999,'hashsize',2^15,...
                          'lambda',1e-3,'eta',1.0,'alpha',0.95,'decay',0.99,'passes',200, ...
                          'mmapprefix','H:','ismacro',false,...
                          'printevery',1,'addorphans',true,...
                          'monfunc', @(res) ...
                            accfunc(res,subxttic,subyttic,xstic,ystic)));
  toc(start)
end

function root=addorphans2(root,orphans,nodeid)
  if (isfield(root,'left'))
    root.left=addorphans2(root.left,orphans,2*nodeid);
    root.right=addorphans2(root.right,orphans,2*nodeid+1);
  else
    root.topy=horzcat(root.topy,find(orphans(:,nodeid))');
  end
end

function root=addorphans(root,orphans)
  root=addorphans2(root,orphans,1);
end

function [yhat,rx]=predict5(res,xtic)
  rx=res.root.route(xtic,false);
  exindex=sparse(1:size(xtic,2),rx,1,size(xtic,2),size(res.root.filtmat,2));

  hashxtic=res.hashmat*xtic;
  yhat=treepredict(hashxtic, ...
                   res.oas.Data.x,...
                   res.root.filtmat, ...
                   exindex, ...
                   res.bias, ...
                   5);
end

function acc=accfunc(res,xttic,yttic,xstic,ystic)
  persistent callnum;

  if (isempty(callnum))
    callnum=0;
  end

  [~,n]=size(xttic);
  [~,m]=size(xstic);
  [c,~]=size(yttic);

  t1=clock;
  [yhatt,routext]=res.predict5(xttic);
  t2=clock;
  [yhats,routexs]=res.predict5(xstic);
  t3=clock;
  trainpredict=n/etime(t2,t1);
  testpredict=m/etime(t3,t2);
  [impweights,avgsomet,maxsomet]=treemakeimpweights(yttic,res.root.filtmat,routext);
  [testimpweights,avgsomes,maxsomes]=treemakeimpweights(ystic,res.root.filtmat,routexs);

  trainfiltprec=sum(impweights)/(5*n);
  nlabelst=full(sum(sum(yttic)));
  trainfiltrecall=sum(impweights)/nlabelst;
  testfiltprec=sum(testimpweights)/(5*m);
  nlabelss=full(sum(sum(ystic)));
  testfiltrecall=sum(testimpweights)/nlabelss;
  trainprec=zeros(1,5);
  testprec=zeros(1,5);
  trainrecall=zeros(1,5);
  testrecall=zeros(1,5);
  traingood=0;
  testgood=0;

  for ii=1:5
    yhatttic=sparse(max(1,yhatt(:,ii)),1:n,1,c,n);
    yhatstic=sparse(max(1,yhats(:,ii)),1:m,1,c,m);
    traingood=traingood+full(sum(dot(yhatttic,yttic,1)));
    testgood=testgood+full(sum(dot(yhatstic,ystic,1)));
    trainprec(ii)=traingood/(ii*n);
    testprec(ii)=testgood/(ii*m);
    trainrecall(ii)=traingood/nlabelst;
    testrecall(ii)=testgood/nlabelss;
  end

  [~,~,st]=find(yhatt);
  uniqt=length(unique(st));
  [~,~,ss]=find(yhats);
  uniqs=length(unique(ss));

  avgdeptht=full(sum(res.root.depthvec(routext)))/n;
  avgdepths=full(sum(res.root.depthvec(routexs)))/m;

  tps=sprintf('%.3g ',trainprec);
  trs=sprintf('%.3g ',trainrecall);
  tf1=sprintf('%.3g ',2./(1./trainprec+1./trainrecall));
  sps=sprintf('%.3g ',testprec);
  srs=sprintf('%.3g ',testrecall);
  sf1=sprintf('%.3g ',2./(1./testprec+1./testrecall));

  fprintf('%u (train) %g %g [%s] [%s] [%s] %.3g %u %.3g %u %g\n', ...
          callnum,trainfiltprec,trainfiltrecall,tps,trs,tf1,...
          avgsomet,maxsomet,avgdeptht,uniqt,trainpredict);
  fprintf('%u (test) %g %g [%s] [%s] [%s] %.3g %u %.3g %u %g\n', ...
          callnum,testfiltprec,testfiltrecall,sps,srs,sf1,...
          avgsomes,maxsomes,avgdepths,uniqs,testpredict);
  callnum=callnum+1;

  acc=testprec(1);
end

function m=makemmapzeros(r,c,format,filename)
  bs=50000;
  n=r*c;
  data=zeros(1,bs,format);
  delete(filename);
  fileID=fopen(filename,'w');
  for off=1:bs:n
    fwrite(fileID,data,format);  % possibly writes extra bytes, but doesn't matter
  end
  fclose(fileID);
  dimensions=[r c];
  clear data;
  m=memmapfile(filename,'Format',{format,dimensions,'x'},'Writable',true);
end

function res=endtoendtree(xtic,ytic,xstic,ystic,params)
  depth=params.depth;
  s=params.s;
  smin=params.smin;
  eta=params.eta;
  decay=params.decay;
  alpha=params.alpha;
  lambda=params.lambda;
  k=params.hashsize;
  passes=params.passes;
  mmapprefix=params.mmapprefix;
  doaddorphans=params.addorphans;
  ismacro=params.ismacro;
  printevery=params.printevery;
  monfunc=params.monfunc;

  start=tic;
  fprintf('hashing data ...');
  [d,~]=size(xtic);
  % double hashing seems to improve a small amount over single hashing
  res.hashmat=sparse([1+mod(randperm(d),k) 1+mod(randperm(d),k)],...
                     [1:d 1:d],...
                     [2*mod(randperm(d),2)-1 2*mod(randperm(d),2)-1]);
  hashxtic=res.hashmat*xtic;
  toc(start)
  fprintf('building tree ...');
  res.root=xhattree(hashxtic,ytic,depth,s,smin,ismacro);
  res.root.route=@(xtic,israndom) treeroute(res.root,res.hashmat*xtic,israndom);
  res.root.filtmat=makefiltmat(res.root,sparse(size(ytic,1),2^(depth+1)));
  rx=res.root.route(xtic,false);
  [impweights,avgsome,maxsome]=treemakeimpweights(ytic,res.root.filtmat,rx);

  [c,n]=size(ytic);
  sumytic=full(sum(ytic,2));
  yimp=ytic*impweights';
  if (doaddorphans)
    orphans=find(yimp==0 & sumytic~=0);
    if (~isempty(orphans))
      t1=clock;
      fprintf('adding %u orphans ...',length(orphans));
      rxindex=sparse(1:n,rx,1,n,2^(depth+1));
      ymap=ytic*rxindex;
      [~,ymaxrx]=max(ymap,[],2);
      yextra=sparse(orphans,ymaxrx(orphans),1,c,2^(depth+1));
      res.root=addorphans(res.root,yextra);
      res.root.route=@(xtic,israndom) treeroute(res.root,res.hashmat*xtic,israndom);
      res.root.filtmat=makefiltmat(res.root,sparse(size(ytic,1),2^(depth+1)));
      rx=res.root.route(xtic,false);
      [impweights,avgsome,maxsome]=treemakeimpweights(ytic,res.root.filtmat,rx);
      t2=clock;
      fprintf('(%g) ',etime(t2,t1));
    end

    yimp=ytic*impweights';
    if (~isempty(find(yimp==0 & sumytic~=0,1)));
      error('wtf');
    end
  end

  res.root.depthvec=makedepthvec(res.root,depth,sparse(2^(depth+1),1));
  res.root.lambdamat=makelambdamat(res.root,sparse(1,2^(depth+1)));

  testrx=res.root.route(xstic,false);
  testimpweights=treemakeimpweights(ystic,res.root.filtmat,testrx);
  toc(start)
  fprintf('orphaned=%u nodes=%u avgsome=%g maxsome=%u avgdepth=%g trainfiltprecat5=%g testfiltprecat5=%g\n',...
          length(find(yimp==0)), ...
          sum(full(sum(res.root.filtmat,1))>0), ...
          avgsome, ...
          maxsome, ...
          full(sum(res.root.depthvec(rx)))/size(rx,1), ...
          sum(impweights)/(5*size(xtic,2)), ...
          sum(testimpweights)/(5*size(xstic,2)));

  [c,~]=size(ytic);

  C=single(max(1,sparseweightedsum(hashxtic,ones(1,n),2)))';
  C=C+lambda*sum(C)/k;

  res.oas=makemmapzeros(k,c,'single',fullfile(mmapprefix,'oas'));
  momentum=makemmapzeros(k,c,'single',fullfile(mmapprefix,'momentum'));
  res.bias=cell(1,2^(depth+1));
  momentumbias=cell(1,2^(depth+1));

  for jj=1:passes
    rx=res.root.route(xtic,true);
    impweights=treemakeimpweights(ytic,res.root.filtmat,rx);

    for ii=unique(rx)'
      candidates=find(res.root.filtmat(:,ii));
      if (isempty(candidates))
        continue;
      end
      if (isempty(res.bias{ii}))
        res.bias{ii}=zeros(length(candidates),1,'single');
        momentumbias{ii}=zeros(length(candidates),1,'single');
      end
    end

    experm=randperm(n);
    exindex=sparse(1:n,rx(experm),impweights(experm),n,size(res.root.filtmat,2));

    nodeperm=randperm(size(res.root.filtmat,2));
    normpreds=treeupdate(hashxtic, ...
                         res.oas.Data.x, ...
                         res.root.filtmat, ...
                         exindex, ...
                         res.bias, ...
                         momentum.Data.x, ...
                         momentumbias, ...
                         ytic, ...
                         C, ...
                         nodeperm, ...
                         experm, ...
                         eta, ...
                         alpha, ...
                         false);

    if (mod(jj,printevery) == 0)
      fprintf('pass: %u normpreds: %g eta:%g ',jj,normpreds/n,eta);
      toc(start)
      res.predict5=@(x) predict5(res,x);
      monfunc(res);
    end
    eta=eta*decay;
  end

  if (mod(jj,printevery) ~= 0)
    fprintf('pass: %u normpreds: %g eta:%g ',jj,normpreds/n,eta);
    toc(start)
    res.predict5=@(x) predict5(res,x);
    monfunc(res);
  end
end

function filtmat=makefiltmat(res,filtmat)
  if (isfield(res,'left'))
    filtmat=makefiltmat(res.left,filtmat);
    filtmat=makefiltmat(res.right,filtmat);
  else
    filtmat(res.topy,res.nodeid)=1;
  end
end

function depthvec=makedepthvec(res,maxdepth,depthvec)
  if (isfield(res,'left'))
    depthvec=makedepthvec(res.left,maxdepth,depthvec);
    depthvec=makedepthvec(res.right,maxdepth,depthvec);
  else
    depthvec(res.nodeid)=maxdepth-res.depth;
  end
end

function lambdamat=makelambdamat(res,lambdamat)
  if (isfield(res,'left'))
    lambdamat=makelambdamat(res.left,lambdamat);
    lambdamat=makelambdamat(res.right,lambdamat);
  else
    lambdamat(res.nodeid)=res.lambda;
  end
end

function b=weightedmedian(w,x)
  [~,ind]=sort(x);
  halftotalw=sum(w)/2;
  cumsumw=cumsum(w(ind));
  bind=find(cumsumw>halftotalw,1,'first');
  b=x(ind(bind));
end

function [w,lambda]=xhat(xtic,ytic,cumulp,mask)
  % eigenvalue of X^\top Y (Y^\top Y)^{-1} Y^\top X
  %
  % which is partial least squares,
  % _but_ using normalized and uncorrelated output variables
  %
  % this is a variant of orthonormal PLS
  % http://aiolos.um.savba.sk/~roman/Papers/pascal05.pdf (equation 15)
  % and also a specific instantion of "canonical ridge analysis"
  % http://aiolos.um.savba.sk/~roman/Papers/pls_book06.pdf (slide 12)

  % NB: this version of xhat is appropriate for multilabel
  %     it will work for multilabel but is not the most efficient
  %     for a faster multilabel specific variant see ../odp/runodp.m

  [d,~]=size(xtic);
  [c,~]=size(ytic);
  p=10;

  xticmask=xtic(:,mask);
  yticmask=ytic(:,mask);

  sumx=sparseweightedsum(xticmask,cumulp(mask),1);
  if (norm(sumx) > 0)
    projx=sumx/norm(sumx);            % (1/|1^\top X|) 1^\top X; projx->dx1
  else
    projx=0;
  end

  % X^\top D Y (Y^\top D Y)^{-1} Y^\top D X

  s=max(sparseweightedsum(yticmask,cumulp(mask),1),min(cumulp(mask)));

  Ztic=randn(p,d,'single');
  % (Y^\top D Y)^{-1} Y^\top D X Omega -> cxp
  Ztic=cheesypcg(@(Z) sparsequad(yticmask,cumulp(mask),yticmask,Z), ...
                 @(Z) bsxfun(@rdivide,Z,s), ...
                 zeros(p,c,'single'), ...
                 sparsequad(yticmask,cumulp(mask),xticmask,Ztic), ...
                 20, ...
                 false);
  % X^\top D Y (Y^\top D Y)^{-1} Y^\top D X Omega
  Ztic=sparsequad(xticmask,cumulp(mask),yticmask,Ztic);
  Ztic=Ztic-(Ztic*projx')*projx;                 % (I - P P^\top)
  [Z,~]=qr(Ztic',0); Ztic=Z'; clear Z;

  % (Y^\top D Y)^{-1} Y^\top D X Omega -> cxp
  Ztic=cheesypcg(@(Z) sparsequad(yticmask,cumulp(mask),yticmask,Z), ...
                 @(Z) bsxfun(@rdivide,Z,s), ...
                 zeros(p,c,'single'), ...
                 sparsequad(yticmask,cumulp(mask),xticmask,Ztic), ...
                 20, ...
                 false);
  % X^\top D Y (Y^\top D Y)^{-1} Y^\top D X Omega
  Ztic=sparsequad(xticmask,cumulp(mask),yticmask,Ztic);
  Ztic=Ztic-(Ztic*projx')*projx;                 % (I - P P^\top)

  [V,S]=eig(Ztic*Ztic');
  [~,Sind]=sort(diag(S),'descend');
  lambda=sqrt(S(Sind(1),Sind(1)));
  w=Ztic'*V(:,Sind(1))/lambda;
end

function res=xhattree(xtic,ytic,depth,s,smin,ismacro)
  if (ismacro)
    numy=max(full(sum(ytic,2)),1)';
  else
    numy=1;
  end
  res=xhattree2(xtic,ytic,depth,s,smin,ones(1,size(xtic,2)),numy,1);
end

function res=xhattree2(xtic,ytic,depth,s,smin,cumulp,numy,nodeid)
  res=struct();
  res.nodeid=nodeid;
  res.lambda=0;
  res.depth=depth;
  res.topy=[];

  eps=1e-4;
  cond=(cumulp>eps);

  if (any(cond))
    sumy=sparseweightedsum(ytic(:,cond),cumulp(cond),1)./numy;
    [~,topy]=sort(sumy,'descend');
    cumuly=cumsum(sumy(topy))/sum(sumy);
    s=min(s,find(cumuly>0.999,1,'first'));
    topy=topy(1:s);
    res.topy=topy;

%     if (depth == 0 && s > smin)
%       fprintf('depth=%u nodeid=%u n=%u s=%u sumy=[%u %u]\n',...
%               depth,nodeid,sum(cumulp(cond)),s,sumy(topy(1)),sumy(topy(end)))
%     end

    if (depth > 0 && s > smin)
      [w,lambda]=xhat(xtic,ytic,cumulp,cond);
      if (lambda > 0) % TODO: figure out "correct" minimum
        thisdp=dmsm(w',xtic(:,cond));
        b=weightedmedian(cumulp(cond),thisdp);
        sigma=sqrt(lambda/sum(cumulp(cond)));
        thisroute=(thisdp-b)/sigma;
        probs=0.5+0.5*erf(thisroute);
        lc=cumulp; lc(cond)=lc(cond).*probs;
        rc=cumulp; rc(cond)=rc(cond).*(1-probs);

        res.wtic=w';
        res.lambda=lambda;
        res.b=b;
        res.sigma=sigma;
        res.left=xhattree2(xtic,ytic,depth-1,s,smin,lc,numy,2*nodeid);
        res.right=xhattree2(xtic,ytic,depth-1,s,smin,rc,numy,2*nodeid+1);
      end
    end
  end
end

% objective is \| A Y - B \|^2
function Y = cheesypcg(Afunc,preAfunc,Y,b,iter,verbose)
  if (isa(b,'single'))
    tol=1e-4;
  else
    tol=1e-6;
  end

  r=b-Afunc(Y); clear b;
  z=preAfunc(r);
  p=z;
  rho=dot(r,z,2); clear z;
  initsumrr=sum(sum(r.*r));
  minY=Y;
  argminY=initsumrr;

  if (verbose)
    fprintf('\n');
  end

  for ii=1:iter
    Ap=Afunc(p);
    alpha=rho./max(dot(p,Ap,2),eps);
    Y=Y+bsxfun(@times,p,alpha);
    deltar=bsxfun(@times,Ap,alpha); clear Ap;
    r=r-deltar;
    newsumrr=sum(sum(r.*r));

    if (newsumrr < argminY)
      minY=Y;
      argminY=newsumrr;
    end

    if (verbose)
      fprintf('iter = %u, newsumrr = %g, initsumrr = %g, relres = %g, argminY = %g\n',ii,newsumrr,initsumrr,newsumrr/initsumrr,argminY);
    end

    if (newsumrr<tol*initsumrr)
        break;
    end

    z=preAfunc(r);
    rho1=-(rho<0).*max(-rho,eps)+(rho>=0).*max(rho,eps);
    rho=-dot(deltar,z,2); % Polak-Ribiere
    beta=rho./rho1;
    p=z+bsxfun(@times,p,beta); clear z;
  end

  if (newsumrr > argminY)
    Y=minY;
  end
end
