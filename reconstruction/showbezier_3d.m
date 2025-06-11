
function inter_pixel_list = showbezier_3d(filename)

modified_str = filename(1:end-4);
cp_filename = [modified_str '_cp.txt'];
res_filename = [modified_str '_res.txt'];
%%
fid=fopen(filename);
data = fscanf(fid, '%g');
c_num = data(1);
count1 = 1;
x=zeros(30000,1);
y=zeros(30000,1);
z=zeros(30000,1);
point_count=0;
for i=1:c_num
    count1 = count1 + 1;
    pixel_num = data(count1);
    for j = 1:pixel_num
        point_count = point_count + 1;
        count1 = count1 + 1;
        x(point_count) = data(count1);
        count1 = count1 + 1;
        y(point_count) = data(count1);
        count1 = count1 + 1;
        z(point_count) = data(count1);       
    end
end
fclose(fid);

%%
b_pixel_num=zeros(2000,1);
b_pixel_sum=zeros(2000,1);
b_x=zeros(2000,200,1);
b_y=zeros(2000,200,1);
b_z=zeros(2000,200,1);
%b_findij=zeros(x_scale,y_scale,2);
fid=fopen(cp_filename);
data = fscanf(fid, '%f');
b_num = data(1);
count = 1;
for i=1:b_num
    count = count + 1;
    b_pixel_num(i) = data(count);
    for j = 1:b_pixel_num(i)
        count = count+ 1;
        b_x(i,j) = data(count);        
        count = count + 1;
        b_y(i,j) = data(count);
        count = count + 1;
        b_z(i,j) = data(count);
    end
end
fclose(fid);

[inter_x,inter_y,inter_z,inter_pixel]=b_3D_interpolation(b_x,b_y,b_z,b_num,b_pixel_num);


fileID=fopen(res_filename,'w+');
for i=1:b_num
    for j=1:inter_pixel(i)
        fprintf(fileID, "%.6f;", inter_x(i,j));
        fprintf(fileID, "%.6f;", inter_y(i,j));
        fprintf(fileID, "%.6f\n", inter_z(i,j));
    end
end
fclose(fileID); 
inter_pixel_list = inter_pixel(1:b_num)';
end

function [inter_x,inter_y,inter_z,inter_pixel]=b_3D_interpolation(b_x,b_y,b_z,b_num,b_pixel_num)
inter_x=zeros(2000,2000);
inter_y=zeros(2000,2000);
inter_z=zeros(2000,2000);
inter_pixel=zeros(2000,1);
%% calculate bezier
for i=1:b_num
    if b_pixel_num(i)>=4
        inter_pixel(i)=inter_pixel(i)+1;
        inter_x(i,inter_pixel(i))=b_x(i,1);
        inter_y(i,inter_pixel(i))=b_y(i,1);
        inter_z(i,inter_pixel(i))=b_z(i,1);
        for j=1:3:b_pixel_num(i)-3
 %           num=pcd*(b2curve(i,j+3)-b2curve(i,j));
            num=50;
            for k=1:num
                u=k*1.0/num;
                b1=(1-u)^3;
                b2=3*u*(1-u)^2;
                b3=3*u^2*(1-u);
                b4=u^3;
                inter_pixel(i)=inter_pixel(i)+1;
                inter_x(i,inter_pixel(i))=b1*b_x(i,j)+b2*b_x(i,j+1)+b3*b_x(i,j+2)+b4*b_x(i,j+3);
                inter_y(i,inter_pixel(i))=b1*b_y(i,j)+b2*b_y(i,j+1)+b3*b_y(i,j+2)+b4*b_y(i,j+3);
                inter_z(i,inter_pixel(i))=b1*b_z(i,j)+b2*b_z(i,j+1)+b3*b_z(i,j+2)+b4*b_z(i,j+3);
            end
        end
    end
end
end
