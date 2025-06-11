
function fitting3D(filename)
warning('off');
errorsquare=0.5;%?????????0.5????????
max_curve_num=10000;
disp(filename);
warning('off');

% ȥ������4���ַ�
modified_str = filename(1:end-4);
% ����µ��ַ�
cp_filename = [modified_str '_cp.txt'];

strc1 = filename;
stro = cp_filename; 

point_count1 = 0;
curve_pixel_num1 = zeros(max_curve_num,1);
curve_pixel_sum1 = zeros(max_curve_num,1);
curve_vx1 = zeros(50000,1);
curve_vy1 = zeros(50000,1);
curve_vz1 = zeros(50000,1);

%% 1
fid=fopen(strc1);
data = fscanf(fid, '%g');
curve_num1 = data(1);
count1 = 1;
for i=1:curve_num1
    count1 = count1 + 1;
    curve_pixel_num1(i) = data(count1);
    for j = 1:curve_pixel_num1(i)
        point_count1 = point_count1 + 1;
        count1 = count1 + 1;
        curve_vx1(point_count1) = data(count1)*1000;
        count1 = count1 + 1;
        curve_vy1(point_count1) = data(count1)*1000;
        count1 = count1 + 1;
        curve_vz1(point_count1) = data(count1)*1000;
    end
end
fclose(fid);
for i=1:point_count1
    curve_vx1(i)=curve_vx1(i)+1;
    curve_vy1(i)=curve_vy1(i)+1;
    curve_vz1(i)=curve_vz1(i)+1;
end
curve_pixel_sum1(1) = curve_pixel_num1(1);
for i = 2:curve_num1
    curve_pixel_sum1(i) = curve_pixel_sum1(i - 1) + curve_pixel_num1(i);
end
%% put in xy list
xy1=zeros(curve_num1,300,2);
for i=1:curve_num1
    for j=1:curve_pixel_num1(i)
        xy1(i,j,1)=curve_vx1(curve_pixel_sum1(i)-curve_pixel_num1(i)+j);
        xy1(i,j,2)=curve_vy1(curve_pixel_sum1(i)-curve_pixel_num1(i)+j);
        xy1(i,j,3)=curve_vz1(curve_pixel_sum1(i)-curve_pixel_num1(i)+j);
    end
end
%%
fid=fopen(stro,'wt');%??????
fprintf(fid,'%g\n',curve_num1);
for i=1:curve_num1
    Mat=zeros(curve_pixel_num1(i),3);
    for j=1:curve_pixel_num1(i)
        Mat(j,1)=xy1(i,j,1);
        Mat(j,2)=xy1(i,j,2);
        Mat(j,3)=xy1(i,j,3);
    end
    ei=curve_pixel_num1(i);
    MxAllowSqD=errorsquare;
    ibi=[1;ei]; %first and last point

    [p0mat,p1mat,p2mat,p3mat,fbi]=bzapproxu(Mat,MxAllowSqD,ibi);
    p0mat=p0mat*0.001;
    p1mat=p1mat*0.001;
    p2mat=p2mat*0.001;
    p3mat=p3mat*0.001;

    [control_count,two]=size(p0mat);
    fprintf(fid,'%g\n',3*control_count+1);
    for j=1:control_count
        fprintf(fid,'%g\t',p0mat(j,1));
        fprintf(fid,'%g\t',p0mat(j,2));
        fprintf(fid,'%g\n',p0mat(j,3));
        fprintf(fid,'%g\t',p1mat(j,1));
        fprintf(fid,'%g\t',p1mat(j,2));
        fprintf(fid,'%g\n',p1mat(j,3));
        fprintf(fid,'%g\t',p2mat(j,1));
        fprintf(fid,'%g\t',p2mat(j,2));
        fprintf(fid,'%g\n',p2mat(j,3));
    end
    fprintf(fid,'%g\t',p3mat(control_count,1));
    fprintf(fid,'%g\t',p3mat(control_count,2));
    fprintf(fid,'%g\n',p3mat(control_count,3));
end
fclose(fid);

end


function Q=bezierInterp(P0,P1,P2,P3,varargin)

%%% Default Values %%%
t=linspace(0,1,101); % uniform parameterization 
defaultValues = {t};
%%% Assign Valus %%%
nonemptyIdx = ~cellfun('isempty',varargin);
defaultValues(nonemptyIdx) = varargin(nonemptyIdx);
[t] = deal(defaultValues{:});
% % --------------------------------
M=[-1  3 -3 1;
    3 -6  3 0;
   -3  3  0 0;
    1  0  0 0];
for k=1:length(t)
    Q(k,:)=[t(k)^3 t(k)^2 t(k) 1]*M*[P0;P1;P2;P3];
end
end

function [MatGlobalInterp]=BezierInterpCPMatSegVec(p0mat,p1mat,p2mat,p3mat,NVec,varargin)

% % % Default Values
ti=[];
defaultValues = {ti};
% % % Assign Valus
nonemptyIdx = ~cellfun('isempty',varargin);
defaultValues(nonemptyIdx) = varargin(nonemptyIdx);
[ti] = deal(defaultValues{:});
% % ---------------------------------------------------------

niarg = nargin; %number of input arguments

MatGlobalInterp=[];
to=0;
firstSegment=1;
for k=1:length(NVec)-1   
    count=NVec(k+1)-NVec(k)+1;
    if(niarg==6)            % if ti is passsed as argument
        from=to+1;
        to  = from+count-1;
        tloc=ti(from:to);   % extracting local t from ti for kth segment 
        
    else                    % ti is not passed, using uniform parameterization
        tloc=linspace(0,1,count);        
    end
    
    %% for two adjacent segments s1 & s2, paremetric value at t=1 for s1
    %% equals t=0 for s2. Therefore no need to evaluate it. Removing t=0 
    %% from tloc from the second segment onwards.
    if (~firstSegment)
        tloc=tloc(2:end);
    end  
    
    MatLocalInterp=bezierInterp( p0mat(k,:),p1mat(k,:),p2mat(k,:),p3mat(k,:),tloc);    
    MatGlobalInterp=[MatGlobalInterp; MatLocalInterp]; % row wise concatenation
    firstSegment=0;
end
end

function [p0mat,p1mat,p2mat,p3mat,fbi,MxSqD]=bzapproxu(Mat,varargin)


p0mat=[];    p1mat=[];    p2mat=[];    p3mat=[];
fbi=[];
MxSqD=0;

if (size(Mat,1) < 4)
    error('Atleast four points are required in Data Matrix');    
end

%%% Default Values %%%
MxAllowSqD=1;
ibi=[1; size(Mat,1)]; % first & last
defaultValues = {MxAllowSqD ibi};
%%% Assign Valus %%%
nonemptyIdx = ~cellfun('isempty',varargin);
defaultValues(nonemptyIdx) = varargin(nonemptyIdx);
[MxAllowSqD ibi] = deal(defaultValues{:});
% % %----------------------------------------
datatype=class(Mat); %original data type 
Mat=double(Mat);   %converte to double (necessary for computation)
MxAllowSqD=double(MxAllowSqD);
% % %----------------------------------------

if(MxAllowSqD<0 )
    error('Max. Allowed Square Distance should be >= 0');    
end

if( ~isvec(ibi) )
    error('arg3 must be row OR column vector');    
end
ibi=getcolvector(ibi);
ibi=[ibi; 1; size(Mat,1)]; % make sure first & last are included
ibi=unique(ibi);           % sort and remove duplicates if any 

[p0mat,p1mat,p2mat,p3mat,ti]=FindBzCP4AllSeg(Mat,ibi,'u');
[MatI]=BezierInterpCPMatSegVec(p0mat,p1mat,p2mat,p3mat,ibi,ti);

[sqDistAry,indexAryGlobal]=MaxSqDistAndInd4EachSegbw2Mat(Mat,MatI, ibi);
sqDistMat=[sqDistAry',indexAryGlobal'];
% localIndex is index of row in sqDistMat that contains MxSqD
[MxSqD, localIndex]=max(sqDistMat(:,1)); 


while(MxSqD > MxAllowSqD)        
    %% appending index of new segmentation into ibi  
    %% index w.r.t Mat where sq. dist. is max among all segments
    MaxSqDistIndex=sqDistMat(localIndex,2); 
    ibi(length(ibi)+1)=MaxSqDistIndex;     % append
    ibi=sort(ibi);                     % sort          
    
    %% Finding range of ibi that would be affected by adding a new
    %% point at max-square-distance postion.
    %% If kth row mataches then get atmost k-1 to k+1 rows of ibi.
    [EffinitialbreaksIndex]=FindGivenRangeMatchedMat([ibi],[1 ; MaxSqDistIndex], 1); 
     
    %% Finding control points of two new segments (obtained by breaking a segment)  
    %% Since we are passing EffinitialbreaksIndex, FindBzCP4AllSeg will only take
    %% relevant segments data from Mat.
    [p0matN,p1matN,p2matN,p3matN,tiN]=FindBzCP4AllSeg(...
                                      Mat,EffinitialbreaksIndex,'u');
    
    %% Combining new and old control point values (old + new + old ).
    %% if only one row in sqDistMat (case when initially there were only two
    %% breakpoints)
    if( size(sqDistMat,1)==1 ) 
        p0mat=p0matN; p1mat=p1matN; p2mat=p2matN; p3mat=p3matN;         
    else 
        p0mat=[p0mat(1:localIndex-1,:); p0matN; p0mat(localIndex+1:end,:)];
        p1mat=[p1mat(1:localIndex-1,:); p1matN; p1mat(localIndex+1:end,:)];
        p2mat=[p2mat(1:localIndex-1,:); p2matN; p2mat(localIndex+1:end,:)];
        p3mat=[p3mat(1:localIndex-1,:); p3matN; p3mat(localIndex+1:end,:)];        
    end     
    
    %% Bezier Interpolatoin to new segments  
    [MatINew]=BezierInterpCPMatSegVec(p0matN,p1matN,p2matN,p3matN,...
                                      EffinitialbreaksIndex,tiN);
    
    si=EffinitialbreaksIndex(1);  % intrp. values ibi(1:si) are already computed
    ei=EffinitialbreaksIndex(end);% intrp. values ibi(ei:end,:) are already computed   
    
    %% Combining new and old interpolation values (old + new + old ).
    %% Not taking common point b/w old-new and b/w new-old
    MatI=[MatI(1:si-1,:); MatINew; MatI(ei+1:end,:)];     
    
    %% now we would find the max-square-distance of affected segments only  
    [sqDistAryN,indexAryGlobalN]=MaxSqDistAndInd4EachSegbw2Mat(...
                                 Mat,MatI, EffinitialbreaksIndex ); % new
    sqDistMatN=[sqDistAryN',indexAryGlobalN'];      % new mat

    %% if only one row in sqDistMat (case when initially
    %% there were only two breakpoints)
    if( size(sqDistMat,1)==1 ) 
        sqDistMat=sqDistMatN;
    else 
    %% combining sqDistMat new and old values (old + new + old)
        sqDistMat=[sqDistMat(1:localIndex-1,:);...
                   sqDistMatN;...
                   sqDistMat(localIndex+1:length(sqDistMat),:)]; 
    end            
    [MxSqD, localIndex]=max(sqDistMat(:,1));     
end

fbi=ibi;
end

function [P0, P1, P2, P3, tout]= FindBezierControlPointsND(p,varargin)

%%% Default Values %%%
ptype='';
defaultValues = {ptype};
%%% Assign Valus %%%
nonemptyIdx = ~cellfun('isempty',varargin);
defaultValues(nonemptyIdx) = varargin(nonemptyIdx);
[ptype] = deal(defaultValues{:});
%%%------------------------------

n=size(p,1);              % number of rows in p

if (strcmpi(ptype,'u') || strcmpi(ptype,'uniform') )
    [t]=linspace(0,1,n);      % uniform parameterized values (normalized b/w 0 to 1)
else
    [t]=ChordLengthNormND(p); % chord-length parameterized values (normalized b/w 0 to 1)
end

P0=p(1,:);       % (at t=0 => P0=p1)
P3=p(n,:);       % (at t=1 => P3=pn)

if (n==1)      % if only one value in p
   P1=P0;      % P1=P0
   P2=P0;      % P2=P0
   
elseif (n==2)  % if only two values in p
   P1=P0;      % P1=P0
   P2=P3;      % P2=P3
   
elseif (n==3)  % if only three values in p
   P1=p(2,:);    % middle point is P1
   P2=p(2,:);    % middle point is P2

else
    
   A1=0;	A2=0;	A12=0;	C1=0;	C2=0; %initialization
    for i=2:n-1 
%    for i=1:n    %it will give same CPs as   i=2:n-1   
      B0 = (1-t(i))^3            ;        % Bezeir Basis
      B1 = ( 3*t(i)*(1-t(i))^2 ) ;
      B2 = ( 3*t(i)^2*(1-t(i)) ) ;
      B3 = t(i)^3                ;
      
      A1  = A1 +  B1^2;
      A2  = A2 +  B2^2;
      A12 = A12 + B1*B2;
      C1 = C1 + B1*( p(i,:) - B0*P0 - B3*P3 );
      C2 = C2 + B2*( p(i,:) - B0*P0 - B3*P3 );
      
   end
   
   DENOM=(A1*A2-A12*A12);       % common denominator for all points
   if(DENOM==0)
       P1=P0;
       P2=P3;
   else
       P1=(A2*C1-A12*C2)/DENOM;
       P2=(A1*C2-A12*C1)/DENOM;
   end
   
end            % END of if-elseif-else conditon

if(nargout==5) % if number of output argument=1 
    tout=t;
end
end

function [p0mat,p1mat,p2mat,p3mat,tout]=FindBzCP4AllSeg(Mat,SegIndexIn,varargin)

%%% Default Values %%%
ptype='';
defaultValues = {ptype};
%%% Assign Valus %%%
nonemptyIdx = ~cellfun('isempty',varargin);
defaultValues(nonemptyIdx) = varargin(nonemptyIdx);
[ptype] = deal(defaultValues{:});
%%%------------------------------

tout=[];
for k=1:length(SegIndexIn)-1
    fromRow=SegIndexIn(k);
    toRow=SegIndexIn(k+1);
    size(Mat(fromRow:toRow,:));
    if (strcmpi(ptype,'u') || strcmpi(ptype,'uniform') )
        [p0,p1,p2,p3,t]= FindBezierControlPointsND(Mat(fromRow:toRow,:),'u'); %uniform parameterization
    else
        [p0,p1,p2,p3,t]= FindBezierControlPointsND(Mat(fromRow:toRow,:));    %chord-length parameterization
    end   

    p0mat(k,:)=p0; 
    p1mat(k,:)=p1;
    p2mat(k,:)=p2;
    p3mat(k,:)=p3;
    tout=horzcat(tout,t);
end
end

function [MatOut]=FindGivenRangeMatchedMat(mat1,mat2,r)
MatOut=[];
k=0;
[r1 c1]=size(mat1);
[r2 c2]=size(mat2);
if (r2~=2)
    disp('Message from FindGivenRangeMatchedMat.m');
    disp('second argument matrix must have two rows');
    return
end

if (c1<c2)
    disp('Message from FindGivenRangeMatchedMat.m');
    disp('numer of columns in second argument matrix must be less than or equal to first argument matrix');
    return
end

for i=1:r1
    flag=1;
  for j=1:c2
      if( mat1(i,mat2(1,j))~=mat2(2,j) )
          flag=0;
          break;
      end      
  end
  if(flag)
      k=i; % found row with values (v1,v2,...vj)
  end
end

if (k~=0)
    x=k-r;
    y=k+r;
    if ( x < 1 )  % backward out of range so we would return from row number 1
        x=1;
    end
    if ( y > r1 ) % forward out of range so we would return until last row
        y=r1;
    end
    MatOut=mat1(x:y,:);       
end
end

function vout=getcolvector(vin)

if(~isvec(vin)  )
    error('input must be a vector');    
end

vout=vin;
[r c]=size(vin);
if (r==1)      % vin is row vector, make it col. vector 
    vout=vin';
end
end

function ans=isvec(x)
ans=0;
d=size(x);

if(length(d)>2) % not vector
    return 
end

[r c]=size(x);

if (r>1 & c>1) % not vector
    return
end

ans=1; % vector
end

function [sqDistAry,indexAryGlobal]=MaxSqDistAndInd4EachSegbw2Mat(mat1,mat2,segIndex)

sqDistAry=[];
indexAryGlobal=[];

for k=1:length(segIndex)-1    
    mat1Seg=mat1(segIndex(k):segIndex(k+1),:);
    mat2Seg=mat2(segIndex(k):segIndex(k+1),:);
    [squaredmaxLocal,indexLocal]=MaxSqDistAndRowIndexbw2Mat(mat1Seg,mat2Seg);
    sqDistAry(k)=squaredmaxLocal;
    indexGlobal=indexLocal+segIndex(k)-1;        
    indexAryGlobal(k)=indexGlobal;
end
end

function [squaredmax,rowIndex]=MaxSqDistAndRowIndexbw2Mat(mat1,mat2)
% mat1=[x11,x12,...x1n;
%       x11,x12,...x1n; 
%       ...
%       xn1,xn2,...xnn; 

% mat2=[y11,y12,...y1n;
%       y11,y12,...y1n; 
%       ...
%       yn1,yn2,...ynn; 
% OR in brief mat1 and mat2 format is like following
%                               [P1;
%                                P2;
%                                P3;
%                                P4;
%                                ...
%                                PN];


if ~isequal(size(mat1),size(mat2))
    error('Two matrices must of of equal size');
end

% %Casting for accurate computation
mat1=double(mat1); 
mat2=double(mat2);

rowIndex=1;
squaredmax=sum( (mat1(1,:)-mat2(1,:)).^2,2 );

for k=1:size(mat1,1)
    % computing square distance b/w kth row
    SqDist=sum( (mat1(k,:)-mat2(k,:)).^2,2 )  ;
    % %  SqDist=TwoNormSqDist(mat1(k,:),mat2(k,:)); %No longer in use
    if(SqDist > squaredmax )
        squaredmax=SqDist;
        rowIndex=k;
    end
end
end

