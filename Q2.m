clear
mu1 = [3, 2];
mu2 = [6, 5];
cov1 = [4, 2; 2, 4];
cov2 = [4, 2; 2, 4];
N = 100;
N_T = [3, 5, 10, 50, 100];
loop = 10;
rng(7)  % For reproducibility
design_data = [mvnrnd(mu1,cov1,N); mvnrnd(mu2,cov2,N)];
design_label = [ones(N,1); zeros(N,1)];
ind = [randsample(N, 100); randsample(N, 100)+N];
train_data = design_data(ind,:);
train_label = design_label(ind);
g_classifer = fitcnb(train_data,train_label);
err_design = mean(abs(train_label - round(predict(g_classifer, train_data))));
err_test = zeros(length(N_T),loop);       

for l =1:loop
    test_data= [mvnrnd(mu1,cov1,N); mvnrnd(mu2,cov2,N)];
    test_label = [ones(N,1); zeros(N,1)];
    for i = 1:length(N_T)
        ind = [];
        for k = 1:N_T(i)
            ind = [ind;randsample(floor(N/N_T(i)), 1)+(k-1)*floor(N/N_T(i))];
            ind = [ind;randsample(floor(N/N_T(i)),1)+(k-1)*floor(N/N_T(i))+N];
        end
%        ind = [randsample(N, N_T(i)); randsample(N, N_T(i))+N];
        test_data_subset = test_data(ind,:);
        test_label_subset = test_label(ind);
        err_test(i,l) = mean(abs(test_label_subset - round(predict(g_classifer, test_data_subset))));
    end
end
err_test_mean = mean(err_test,2);
err_test_var = var(err_test,0,2);
figure(1)
hold on
plot(N_T, err_test_mean, 'rx-','Linewidth',2,'Markersize',8)
plot(N_T, err_test_var, 'bx-','Linewidth',2,'Markersize',8)
xlabel("Size of Test Set")
legend("Mean of E_{test}","Variance of E_{test}")

test_data= [mvnrnd(mu1,cov1,N); mvnrnd(mu2,cov2,N)];
test_label = [ones(N,1); zeros(N,1)];
N_D = [3, 5, 10, 50, 75, 90];
N_T = N - N_D;
err_design = zeros(length(N_D),loop);   
err_test = zeros(length(N_D),loop);   
for i = 1:length(N_D)
    for l =1:loop
        ind_train = [randsample(N, N_D(i)); randsample(N, N_D(i))+N];
        ind_train = [];
        for k = 1:N_D(i)
            ind_train = [ind_train;randsample(floor(N/N_D(i)), 1)+(k-1)*floor(N/N_D(i))];
            ind_train = [ind_train;randsample(floor(N/N_D(i)),1)+(k-1)*floor(N/N_D(i))+N];
        end
        ind_test = [randsample(N, N_T(i)); randsample(N, N_T(i))+N];
        ind = [];
        for k = 1:N_T(i)
            ind_test = [ind_test;randsample(floor(N/N_T(i)), 1)+(k-1)*floor(N/N_T(i))];
            ind_test = [ind_test;randsample(floor(N/N_T(i)),1)+(k-1)*floor(N/N_T(i))+N];
        end
        train_data = design_data(ind_train,:);
        train_label = design_label(ind_train);
        test_data_subset = test_data(ind_test,:);
        test_label_subset = test_label(ind_test);
        err_design(i,l) = mean(abs(train_label - round(predict(g_classifer, train_data))));
        err_test(i,l) = mean(abs(test_label_subset - round(predict(g_classifer, test_data_subset))));
    end
end
err_design_mean = mean(err_design,2);
std(err_design,0,2)
err_test_mean = mean(err_test,2);
err_result = [err_design_mean, err_test_mean]