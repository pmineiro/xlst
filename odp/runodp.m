function res=runodp()
  [mypath,~,~]=fileparts(mfilename('fullpath'));
  addpath(fullfile(mypath,'..','matlab'));
  
  % change this if you want, e.g., to an SSD drive location
  mmapprefix='d:\d_drive\';

  randn('seed',90210);
  rand('seed',8675309);
  
  start=tic;
  fprintf('loading data ... ');

  if (exist('dmsm') ~= 3 || ...
      exist('sparsequad') ~= 3 || ... 
      exist('sparseweightedsum') ~= 3 || ...
      exist('treemakeimpweights') ~= 3 || ...
      exist('treepredict') ~= 3 || ...
      exist('treeroute') ~= 3 || ...
      exist('treeupdate') ~= 3)
    error('you must compile the mex.  go to ../matlab and type ''make'' at the command line.');
  end
  
  if (exist(fullfile(cd,'odpmunge.mat'),'file') == 2)
    load('odpmunge.mat');
  else
    error('you must download odpmunge.mat, available at http://1drv.ms/1MTF7A1');
  end
  toc(start)
  
  [~,n]=size(xttic);

  subn=ceil(0.1*n);
  subtrain=sort(randperm(n,subn));
  subxttic=xttic(:,subtrain);
  subyttic=yttic(:,subtrain);
  
  start=tic;
  res=endtoendtree(xttic,yttic,xstic,ystic,...
                   struct('depth',14,'s',4000,'smin',3999,'hashsize',2^15,...
                          'lambda',1e-3,'eta',1.0,'alpha',0.95,'decay',0.99,'passes',200, ...
                          'mmapprefix',mmapprefix,'printevery',1,...
                          'monfunc', @(res) ...
                            accfunc(res,subxttic,subyttic,xstic,ystic)));
  toc(start)
end

function [yhat,rx]=predict(res,xtic)
  rx=res.root.route(xtic,false);
  exindex=sparse(1:size(xtic,2),rx,1,size(xtic,2),size(res.root.filtmat,2));

  hashxtic=res.hashmat*xtic;
  yhat=treepredict(hashxtic, ...
                   res.oas.Data.x,... 
                   res.root.filtmat, ...
                   exindex, ...
                   res.bias, ...
                   1);
end

function acc=accfunc(res,xttic,yttic,xstic,ystic)
  persistent callnum;
  
  if (isempty(callnum))
    callnum=0;
  end
  
  [~,n]=size(xttic);
  t1=clock;
  [yhat,rx]=res.predict(xttic);
  t2=clock;
  trainpredict=n/etime(t2,t1);
  [~,truey]=max(yttic,[],1); truey=truey';
  [impweights,avgsomet]=treemakeimpweights(yttic,res.root.filtmat,rx);
  filtacc=sum(impweights>0)/n;
  acc=sum(yhat==truey)/n;
  uniqt=length(unique(yhat));
  dv=res.root.depthvec(rx);
  avgdeptht=sum(dv)/size(xttic,1);
  
  [~,m]=size(xstic);
  t1=clock;
  [yhats,rx]=res.predict(xstic);
  t2=clock;
  testpredict=m/etime(t2,t1);
  [~,trueys]=max(ystic,[],1); trueys=trueys';
  [impweights,avgsomes]=treemakeimpweights(ystic,res.root.filtmat,rx);
  testfiltacc=sum(impweights>0)/m;
  testacc=sum(yhats==trueys)/m;
  uniqs=length(unique(yhats));
  dv=res.root.depthvec(rx);
  avgdepths=sum(dv)/size(xstic,2);
  
  fprintf('%u acc = (train) %g %g %.3g %.3g %u %g (test) %g %g %.3g %.3g %u %g\n',...
          callnum,filtacc,acc,avgsomet,avgdeptht,uniqt,trainpredict,...
                  testfiltacc,testacc,avgsomes,avgdepths,uniqs,testpredict);
  callnum=callnum+1;
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
  printevery=params.printevery;
  monfunc=params.monfunc;

  start=tic;
  fprintf('building tree ...');
  res.root=xhattree(xtic,ytic,depth,s,smin);
  res.root.route=@(xtic,israndom) treeroute(res.root,xtic,israndom);
  res.root.filtmat=makefiltmat(res.root,sparse(size(ytic,1),2^(depth+1)));
  res.root.depthvec=makedepthvec(res.root,depth,zeros(2^(depth+1),1));
  rx=res.root.route(xtic,false);
  [impweights,avgsome]=treemakeimpweights(ytic,res.root.filtmat,rx);
  testrx=res.root.route(xstic,false);
  testimpweights=treemakeimpweights(ystic,res.root.filtmat,testrx);
  toc(start)
  fprintf('orphaned=%u nodes=%u avgsome=%g avgdepth=%g trainfiltacc=%g testfiltacc=%g\n',...
          sum(full(sum(res.root.filtmat,2))==0), ...
          sum(full(sum(res.root.filtmat,1))>0), ...
          avgsome, ...
          sum(res.root.depthvec(rx))/size(rx,1), ...
          sum(impweights>0)/size(xtic,2), ...
          sum(testimpweights>0)/size(xstic,2));
              
  [d,n]=size(xtic);
  [c,~]=size(ytic);
  % double hashing seems to improve a small amount over single hashing
  res.hashmat=sparse([1+mod(randperm(d),k) 1+mod(randperm(d),k)],... 
                     [1:d 1:d],...
                     [2*mod(randperm(d),2)-1 2*mod(randperm(d),2)-1]);

  hashxtic=res.hashmat*xtic;                 
                 
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
                         true);

    if (mod(jj,printevery) == 0)
      fprintf('pass: %u normpreds: %g eta:%g ',jj,normpreds/n,eta);
      toc(start)
      res.predict=@(x) predict(res,x);
      monfunc(res);
    end
    eta=eta*decay;
  end

  if (mod(jj,printevery) ~= 0)
    fprintf('pass: %u normpreds: %g eta:%g ',jj,normpreds/n,eta);
    toc(start)
    res.predict=@(x) predict(res,x);
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

  p=10;
  
  xticmask=xtic(:,mask);
  yticmask=ytic(:,mask);
  [d,n]=size(xticmask);
  
  sumx=sparseweightedsum(xticmask,cumulp(mask),1);
  if (norm(sumx) > 0)
    projx=sumx/norm(sumx);            % (1/|1^\top X|) 1^\top X; projx->dx1
  else
    projx=0;
  end
  
  s=max(sparseweightedsum(yticmask,cumulp(mask),1),1);
  [~,yhat]=max(yticmask,[],1); yhat=yhat';
  scale=bsxfun(@rdivide,cumulp(mask),s(yhat));

  Ztic=randn(p,d,'single');
                                                        %     cxc           cxn  nxd dxp
  Ztic=sparsequad(yticmask,scale,xticmask,Ztic);        % (Y^\top D Y)^{-1} Y^\top X Omega
  Ztic=sparsequad(xticmask,cumulp(mask),yticmask,Ztic); % X^\top Y (Y^\top D Y)^{-1} Y^\top X Omega
  Ztic=Ztic-(Ztic*projx')*projx;                        % (I - P P^\top)
  [Z,~]=qr(Ztic',0); Ztic=Z'; clear Z;  
  
  Ztic=sparsequad(yticmask,scale,xticmask,Ztic);        % (Y^\top D Y)^{-1} Y^\top X Omega
  Ztic=sparsequad(xticmask,cumulp(mask),yticmask,Ztic); % X^\top Y (Y^\top D Y)^{-1} Y^\top X Omega
  Ztic=Ztic-(Ztic*projx')*projx;                        % (I - P P^\top)

  [V,S]=eig(Ztic*Ztic');
  [~,Sind]=sort(diag(S),'descend');
  lambda=sqrt(S(Sind(1),Sind(1)));
  w=Ztic'*V(:,Sind(1))/lambda;
end

function res=xhattree(xtic,ytic,depth,s,smin)
  res=xhattree2(xtic,ytic,depth,s,smin,ones(1,size(xtic,2)),1);
end

function res=xhattree2(xtic,ytic,depth,s,smin,cumulp,nodeid)
  res=struct();
  res.nodeid=nodeid;
  res.lambda=0;
  res.depth=depth;
  res.topy=[];

  eps=1e-4;
  cond=(cumulp>eps);
  
  if (any(cond))
    sumy=sparseweightedsum(ytic(:,cond),cumulp(cond),1);
    [~,topy]=sort(sumy,'descend');
    cumuly=cumsum(sumy(topy))/sum(sumy);
    s=min(s,find(cumuly>0.999,1,'first'));
    topy=topy(1:s);
    res.topy=topy;
  
    if (depth > 0 && s > smin)
      [w,lambda]=xhat(xtic,ytic,cumulp,cond);
      if (lambda > 0)
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
        res.left=xhattree2(xtic,ytic,depth-1,s,smin,lc,2*nodeid);
        res.right=xhattree2(xtic,ytic,depth-1,s,smin,rc,2*nodeid+1);
      end
    end
  end
end
