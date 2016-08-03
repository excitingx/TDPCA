clc
clear
%%%%%%%%%%%%%--------------2D PCA特征提取，并用1NN进行分类--------------%%%%%%%%%%%%%

%load face database
load('orldata.mat');
facedatabase = double(orldata);
nclass = 40;    % 样本中有40个人
nsample_eachclass = 10;     % 每个人10张图
neachtrain = 5;     % 每个人取5张做训练样本
neachtest = 5;      % 每个人取5张做测试样本
height = 112;       % 图的高
width = 92;     % 图的宽

%------------------训练数据集，height*width*num的矩阵，trainingSet------------------
trainingSet = zeros(height,width,neachtrain*nclass);
%将原始数据集中的一维向量转换为二维矩阵，放入三维矩阵中，第三维表示第几个样本
for i = 1:nclass
    for j = 1:neachtrain
        %trainingVet(:,(i-1)*neachtrain+j) = facedatabase(:,(i-1)*nsample_eachclass+j*2-1);
        trainingSet(:,:,(i-1)*neachtrain+j) = reshape(facedatabase(:,(i-1)*nsample_eachclass+j),height,width);
    end
end

%------------------测试数据，height*width*num的矩阵，testingSet------------------
testingSet = zeros(height,width,neachtest*nclass);
for i = 1:nclass
    for j = 1:neachtest
        testingSet(:,:,(i-1)*neachtest+j) = reshape(facedatabase(:,(i-1)*nsample_eachclass+5+j),height,width);
    end
end

numClass = 40;      % 训练样本和测试样本所拥有的类别数numClass

%------------------2DPCA降维特征提取并1NN分类------------------
accuracy = zeros(1,10);
for thresthold = 2:2:20
    [accuracy(thresthold/2),right,numShape] = PCA_2D_Classifier(trainingSet,testingSet,numClass,thresthold);
end
accuracy

box
axis([2 20 0.65 1])
set(gca,'xtick',2:2:20,'ytick',0.65:0.05:1)
ylabel('识别率');
xlabel('特征数');

hold on;
plot(2:2:20,accuracy,'-^')


