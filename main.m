clc
clear
%%%%%%%%%%%%%--------------2D PCA������ȡ������1NN���з���--------------%%%%%%%%%%%%%

%load face database
load('orldata.mat');
facedatabase = double(orldata);
nclass = 40;    % ��������40����
nsample_eachclass = 10;     % ÿ����10��ͼ
neachtrain = 5;     % ÿ����ȡ5����ѵ������
neachtest = 5;      % ÿ����ȡ5������������
height = 112;       % ͼ�ĸ�
width = 92;     % ͼ�Ŀ�

%------------------ѵ�����ݼ���height*width*num�ľ���trainingSet------------------
trainingSet = zeros(height,width,neachtrain*nclass);
%��ԭʼ���ݼ��е�һά����ת��Ϊ��ά���󣬷�����ά�����У�����ά��ʾ�ڼ�������
for i = 1:nclass
    for j = 1:neachtrain
        %trainingVet(:,(i-1)*neachtrain+j) = facedatabase(:,(i-1)*nsample_eachclass+j*2-1);
        trainingSet(:,:,(i-1)*neachtrain+j) = reshape(facedatabase(:,(i-1)*nsample_eachclass+j),height,width);
    end
end

%------------------�������ݣ�height*width*num�ľ���testingSet------------------
testingSet = zeros(height,width,neachtest*nclass);
for i = 1:nclass
    for j = 1:neachtest
        testingSet(:,:,(i-1)*neachtest+j) = reshape(facedatabase(:,(i-1)*nsample_eachclass+5+j),height,width);
    end
end

numClass = 40;      % ѵ�������Ͳ���������ӵ�е������numClass

%------------------2DPCA��ά������ȡ��1NN����------------------
accuracy = zeros(1,10);
for thresthold = 2:2:20
    [accuracy(thresthold/2),right,numShape] = PCA_2D_Classifier(trainingSet,testingSet,numClass,thresthold);
end
accuracy

box
axis([2 20 0.65 1])
set(gca,'xtick',2:2:20,'ytick',0.65:0.05:1)
ylabel('ʶ����');
xlabel('������');

hold on;
plot(2:2:20,accuracy,'-^')


