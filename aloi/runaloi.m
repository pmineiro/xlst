% quick roadmap:
%
% the routine xhat() corresponds to the eigenvalue problem associated
% with learning a tree node

function res=runaloi()
  [mypath,~,~]=fileparts(mfilename('fullpath'));
  addpath(fullfile(mypath,'..','matlab'));

  randn('seed',90210);
  rand('seed',8675309);

  tic
  fprintf('loading data ...');
  load(fullfile(mypath,'aloi.mat'));
  [n,~]=size(xt);
  [m,~]=size(xs);
  toc

  xttic=xt';
  yttic=yt';
  xstic=xs';
  ystic=ys';

  tic
  dbstop if naninf
  res=endtoendtree(xttic,yttic,xstic,ystic,...
                   struct('depth',12,'s',25,'smin',24,'rank',50,...
                          'lambda',1e-2,'eta',0.25,'alpha',0.8,'decay',0.999,'passes',200, ...
                          'monfunc', @(res) ...
                            true)); %accfunc(res,xttic,yttic,xstic,ystic)));
  toc
  accfunc(res,xttic,yttic,xstic,ystic);
  [yhatt,routext]=res.predict(xttic);
  [yhats,routexs]=res.predict(xstic);
  impweights=makeimpweights(yttic,res.root.filtmat,routext);
  testimpweights=makeimpweights(ystic,res.root.filtmat,routexs);
  res.trainfiltacc=sum(impweights)/n;
  res.testfiltacc=sum(testimpweights)/m;
  [~,trainy]=max(yttic,[],1); trainy=trainy';
  [~,testy]=max(ystic,[],1); testy=testy';
  res.trainacc=sum(yhatt==trainy)/n;
  res.testacc=sum(yhats==testy)/m;
end

function where=route2(res,xtic,mask,where,israndom)
  if (~isempty(mask))
    if (isfield(res,'left'))
      thisdp=res.wtic*xtic(:,mask);
      if (israndom)
        thisroute=(thisdp-res.b)/res.sigma;
        probs=0.5+0.5*erf(thisroute);
        thisbit=(rand(size(probs))<probs);
      else
        thisbit=thisdp>res.b;
      end
      where=route2(res.left,xtic,mask(thisbit),where,israndom);
      where=route2(res.right,xtic,mask(~thisbit),where,israndom);
    else
      where(mask)=res.nodeid;
    end
  end
end

function where=route(res,xtic,israndom)
  where=route2(res,xtic,1:size(xtic,2),zeros(size(xtic,2),1),israndom);
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

function [yhat,rx]=predict(res,xtic)
  rx=res.root.route(xtic,false);
  yhat=zeros(size(xtic,2),1);
  exindex=sparse(1:length(rx),rx,1);

  for ii=unique(rx)'
    candidates=find(res.root.filtmat(:,ii));
    exs=find(exindex(:,ii));
    if (isempty(candidates) || isempty(exs))
      continue;
    end
    xticgood=xtic(:,exs);
    embedtic=res.oasone'*xticgood;                              % k-by-ex

    if (~isempty(res.bias{ii}))
      preds=embedtic'*res.oastwo{ii};
      preds=bsxfun(@plus,preds,res.bias{ii});
      [~,thisyhat]=max(preds,[],2);
      yhat(exs)=candidates(thisyhat);
    end
  end
end

function [impweights,avgsome,maxsome]=makeimpweights(ytic,filtmat,rx)
  n=length(rx);
  impweights=zeros(1,n);
  avgsome=0;
  maxsome=0;
  bs=50000;
  for off=1:bs:n
    offend=min(n,off+bs-1);
    fm=filtmat(:,rx(off:offend));
    impweights(off:offend)=full(dot(ytic(:,off:offend),fm,1));
    avgsome=avgsome+full(sum(sum(fm)));
    maxsome=max(maxsome,full(max(sum(fm))));
  end
  avgsome=avgsome/n;
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
  [impweights,avgsomet,maxsomet]=makeimpweights(yttic,res.root.filtmat,rx);
  filtacc=sum(impweights>0)/n;
  acc=sum(yhat==truey)/n;
  uniqt=length(unique(yhat));
  dv=res.root.depthvec(rx);
  avgdeptht=full(sum(dv))/size(xttic,1);

  [~,m]=size(xstic);
  t1=clock;
  [yhats,rx]=res.predict(xstic);
  t2=clock;
  testpredict=m/etime(t2,t1);
  [~,trueys]=max(ystic,[],1); trueys=trueys';
  [impweights,avgsomes,maxsomes]=makeimpweights(ystic,res.root.filtmat,rx);
  testfiltacc=sum(impweights>0)/m;
  testacc=sum(yhats==trueys)/m;
  uniqs=length(unique(yhats));
  dv=res.root.depthvec(rx);
  avgdepths=full(sum(dv))/size(xstic,2);

  fprintf('%u acc = (train) %g %g %.3g %u %.3g %u %g (test) %g %g %.3g %u %.3g %u %g\n',...
          callnum,filtacc,acc,avgsomet,maxsomet,avgdeptht,uniqt,trainpredict,...
                  testfiltacc,testacc,avgsomes,maxsomes,avgdepths,uniqs,testpredict);
  callnum=callnum+1;
end

function res=endtoendtree(xtic,ytic,xstic,ystic,params)
  depth=params.depth;
  s=params.s;
  smin=params.smin;
  eta=params.eta;
  decay=params.decay;
  alpha=params.alpha;
  lambda=params.lambda;
  k=params.rank;
  passes=params.passes;
  monfunc=params.monfunc;

  start=tic;
  fprintf('building tree ...');
  res.root=xhattree(xtic,ytic,depth,s,smin);
  res.root.route=@(xtic,israndom) route(res.root,xtic,israndom);
  res.root.filtmat=makefiltmat(res.root,sparse(size(ytic,1),2^(depth+1)));
  rx=res.root.route(xtic,false);
  [impweights,avgsome,maxsome]=makeimpweights(ytic,res.root.filtmat,rx);

  [c,n]=size(ytic);
  rxindex=sparse(1:n,rx,1,n,2^(depth+1));
  ymap=ytic*rxindex;
  [~,ymaxrx]=max(ymap,[],2);
  yimp=ytic*impweights';
  orphans=find(yimp==0);
  if (~isempty(orphans))
    t1=clock;
    fprintf('adding %u orphans ...',length(orphans));
    yextra=sparse(orphans,ymaxrx(orphans),1,c,2^(depth+1));
    res.root=addorphans(res.root,yextra);
    res.root.route=@(xtic,israndom) route(res.root,xtic,israndom);
    res.root.filtmat=makefiltmat(res.root,sparse(size(ytic,1),2^(depth+1)));
    rx=res.root.route(xtic,false);
    [impweights,avgsome,maxsome]=makeimpweights(ytic,res.root.filtmat,rx);
    t2=clock;
    fprintf('(%g) ',etime(t2,t1));
  end

  res.root.depthvec=makedepthvec(res.root,depth,sparse(2^(depth+1),1));
  res.root.lambdamat=makelambdamat(res.root,sparse(1,2^(depth+1)));

  testrx=res.root.route(xstic,false);
  testimpweights=makeimpweights(ystic,res.root.filtmat,testrx);
  toc(start)
  fprintf('orphaned=%u nodes=%u avgsome=%g maxsome=%u avgdepth=%g trainfiltacc=%g testfiltacc=%g\n',...
          sum(full(sum(res.root.filtmat,2))==0), ...
          sum(full(sum(res.root.filtmat,1))>0), ...
          avgsome, ...
          maxsome, ...
          full(sum(res.root.depthvec(rx)))/size(rx,1), ...
          sum(impweights>0)/size(xtic,2), ...
          sum(testimpweights>0)/size(xstic,2));

  [d,n]=size(xtic);
  [c,~]=size(ytic);
  res.oasone=randn(d,k)/sqrt(d);
  res.oastwo=cell(1,2^(depth+1));
  res.bias=cell(1,2^(depth+1));
  momentumone=zeros(d,k);
  momentumtwo=cell(1,2^(depth+1));
  momentumbias=cell(1,2^(depth+1));

  megatrace=sum(sum(xtic.*xtic));
  C=full(0.5*(xtic*xtic')+lambda*megatrace*speye(d)/d);
  C=chol(C,'lower');

  for jj=1:passes
    rx=res.root.route(xtic,true);
    uniqrx=unique(rx);
    exindex=sparse(1:n,rx,makeimpweights(ytic,res.root.filtmat,rx));

    normpreds=0;
    for ii=randperm(length(uniqrx))
      goodexs=find(exindex(:,uniqrx(ii)));
      candidates=find(res.root.filtmat(:,uniqrx(ii)));
      if (isempty(candidates) || isempty(goodexs))
        continue;
      end
      if (isempty(res.bias{uniqrx(ii)}))
        res.bias{uniqrx(ii)}=zeros(1,length(candidates));
        momentumbias{uniqrx(ii)}=zeros(1,length(candidates));
        res.oastwo{uniqrx(ii)}=zeros(k,length(candidates));
        momentumtwo{uniqrx(ii)}=zeros(k,length(candidates));
      end

      xticgood=xtic(:,goodexs);
      embedtic=res.oasone'*xticgood;                         % k-by-ex
      preds=embedtic'*res.oastwo{uniqrx(ii)};                % ex-by-cand
      preds=bsxfun(@plus,preds,res.bias{uniqrx(ii)});
      preds=exp(bsxfun(@minus,preds,max(preds,[],2)));
      preds=bsxfun(@rdivide,preds,sum(preds,2));
      preds=preds-ytic(candidates,goodexs)';
      normpreds=normpreds+sum(sum(preds.*preds));

      gtwo=embedtic*preds;                                   % k-by-cand

      deltaone=preds*res.oastwo{uniqrx(ii)}';                % ex-by-k
      gone=xticgood*deltaone;                                % d-by-k
      usedF=any(xticgood,2); gone(usedF,:)=C(usedF,usedF)'\(C(usedF,usedF)\gone(usedF,:));
      momentumone=alpha*momentumone-eta*gone;
      res.oasone=res.oasone+momentumone;
      nex=length(goodexs);
      momentumtwo{uniqrx(ii)}=alpha*momentumtwo{uniqrx(ii)}-(eta/nex)*gtwo;
      res.oastwo{uniqrx(ii)}=res.oastwo{uniqrx(ii)}+momentumtwo{uniqrx(ii)};
      momentumbias{uniqrx(ii)}=alpha*momentumbias{uniqrx(ii)}-eta*mean(preds);
      res.bias{uniqrx(ii)}=res.bias{uniqrx(ii)}+momentumbias{uniqrx(ii)};
    end
    if (mod(jj,50) == 0)
      fprintf('pass: %u normpreds: %g eta:%g\n',jj,normpreds/n,eta);
      res.predict=@(x) predict(res,x);
      monfunc(res);
    end
    eta=eta*decay;
  end

  if (mod(jj,50) ~= 0)
    fprintf('pass: %u normpreds: %g eta:%g\n',jj,normpreds/n,eta);
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

  p=10;

  xticmask=xtic(:,mask);
  yticmask=ytic(:,mask);
  [d,n]=size(xticmask);

  sumx=sum(bsxfun(@times,xticmask,cumulp(mask)),2)';
  if (norm(sumx) > 0)
    projx=sumx/norm(sumx);            % (1/|1^\top X|) 1^\top X; projx->dx1
  else
    projx=0;
  end

  s=max(sum(bsxfun(@times,yticmask,cumulp(mask)),2),1);
  [~,yhat]=max(yticmask,[],1); yhat=yhat';
  scale=bsxfun(@rdivide,cumulp(mask)',s(yhat));

  Z=randn(d,p);
                                                      %     cxc           cxn  nxd dxp
  Z=yticmask*bsxfun(@times,xticmask'*Z,scale);        % (Y^\top D Y)^{-1} Y^\top D X Omega
  Z=xticmask*bsxfun(@times,yticmask'*Z,cumulp(mask)');% X^\top D Y (Y^\top D Y)^{-1} Y^\top X Omega
  Z=Z-projx'*(projx*Z);                               % (I - P P^\top)
  [Z,~]=qr(Z,0); 

  Z=yticmask*bsxfun(@times,xticmask'*Z,scale);        % (Y^\top D Y)^{-1} Y^\top D X Omega
  Z=xticmask*bsxfun(@times,yticmask'*Z,cumulp(mask)');% X^\top D Y (Y^\top D Y)^{-1} Y^\top X Omega
  Z=Z-projx'*(projx*Z);                               % (I - P P^\top)

  [V,S]=eig(Z'*Z);
  [~,Sind]=sort(diag(S),'descend');
  lambda=sqrt(S(Sind(1),Sind(1)));
  w=Z*V(:,Sind(1))/lambda;
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
    sumy=sum(bsxfun(@times,ytic(:,cond),cumulp(cond)),2)';
    [~,topy]=sort(sumy,'descend');
    cumuly=cumsum(sumy(topy))/sum(sumy);
    s=min(s,find(cumuly>0.999,1,'first'));
    topy=topy(1:s);
    res.topy=topy;

    if (depth > 0 && s > smin)
      [w,lambda]=xhat(xtic,ytic,cumulp,cond);
      if (lambda > 0) 
        thisdp=w'*xtic(:,cond);
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
