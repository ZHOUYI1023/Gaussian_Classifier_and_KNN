mu1 = [3, 2];
mu2 = [6, 5];
cov1 = [4, 2; 2, 4];
cov2 = [4, 2; 2, 4];
N = 100;
N_D = [3, 5, 10, 50, 100];
loop = 10;
rng(3)  % For reproducibility
design_data = [mvnrnd(mu1,cov1,N); mvnrnd(mu2,cov2,N)];
design_label = [ones(N,1); zeros(N,1)];
test_data = [mvnrnd(mu1,cov1,N); mvnrnd(mu2,cov2,N)];
test_label = [ones(N,1); zeros(N,1)];
figure(1)
hold on
plot(design_data(1:N,1),design_data(1:N,2),'x-','Linewidth',2,'Markersize',8)
plot(design_data(N+1:2*N,1),design_data(N+1:2*N,2),'+-','Linewidth',2,'Markersize',8)
legend("class 1","class 2")
err_design = zeros(length(N_D), loop);
err_test = zeros(length(N_D), loop);
for i = 1:length(N_D)
    for j = 1:loop
        ind = [randsample(N, N_D(i)); randsample(N, N_D(i))+N];
        train_data = design_data(ind,:);
        train_label = design_label(ind);
        g_classifer = fitcnb(train_data,train_label);
        err_design(i,j) = mean(abs(train_label - round(predict(g_classifer, train_data))));
        err_test(i,j) = mean(abs(test_label - round(predict(g_classifer, test_data))));
%         if sum(train_label) < 2
%             err_design(i, j) = 0;
%             err_test(i, j) = mean(abs(test_label - 0));
%         elseif sum(train_label) > 2*N_D(i) - 2
%             err_design(i, j) = 0;
%             err_test(i, j) = mean(abs(test_label - 1));
%         else
%             g_classifer = fitcnb(train_data,train_label);
%             err_design(i,j) = mean(abs(train_label - round(predict(g_classifer, train_data))));
%             err_test(i,j) = mean(abs(test_label - round(predict(g_classifer, test_data))));
%         end
    end
end
err_design_mean = mean(err_design,2);
err_test_mean = mean(err_test,2);
figure(2)
hold on
plot(N_D, err_design_mean, 'x-','Linewidth',2,'Markersize',8)
plot(N_D, err_test_mean, 'x-','Linewidth',2,'Markersize',8)

err_knn_min = 2*N*ones(length(N_D),1);
std_knn = 2*N*ones(length(N_D),1);
k_min = zeros(length(N_D),1);
for i = 1:length(N_D)
    for k = 1:2:min(51, 2*N_D(i) -1)
        err_knn = zeros(loop,1);
        for j = 1: loop
            ind = [randsample(N, N_D(i)); randsample(N, N_D(i))+N];
            train_data = design_data(ind,:);
            train_label = design_label(ind);
            knn_classifier = fitcknn(train_data,train_label,'NumNeighbors',k);
            err_knn(j) = mean(abs(test_label - predict(knn_classifier, test_data)));
        end
        if mean(err_knn) < err_knn_min(i)
            err_knn_min(i) = mean(err_knn);
            k_min(i) = k;
            std_knn(i) = std(err_knn,0,1);
        end
    end
end
figure(2)
hold on
plot(N_D, err_knn_min, 'x-','Linewidth',2,'Markersize',8)
xlabel("Size of Training Set")
ylabel("Average Error")
legend("Gaussian Classifer E_{design}","Gaussian Classifer E_{test}", "KNN E_{test}")
figure
plot(N_D, k_min, 'x-','Linewidth',2,'Markersize',8)
xlabel("Size of Training Set")
ylabel("K")
figure
plot(N_D, std_knn, 'x-','Linewidth',2,'Markersize',8)
xlabel("Size of Training Set")
ylabel("Standard Deviation")