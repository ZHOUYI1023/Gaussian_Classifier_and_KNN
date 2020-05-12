clear
N_D = [3, 5, 10, 20, 50, 100, 200];
d = [5, 10, 15];
loop = 10;
N = 500;
err_test_mean = zeros(length(d),length(N_D));
err_test = zeros(length(N_D),loop);  
err_true = zeros(length(d),1);
for i = 1:length(d)
%     if(i == 1)
%         mu1 = (2.15-0.3+0.045+0.1241-0.4)*rand(1,d(i));%[2.5, zeros(1, d(i)-1)];
%     elseif(i == 2)
%         mu1 = (2.15-0.3+0.05+0.148-0.3985)*rand(1,d(i));
%     elseif(i ==3)
%         mu1 = (2.15-3*0.3+0.04-0.19)*rand(1,d(i));
%     end
if(i == 1)
        mu1 = (2.15-0.3+0.045)*rand(1,d(i));%[2.5, zeros(1, d(i)-1)];
    elseif(i == 2)
        mu1 = (2.15-0.3+0.05)*rand(1,d(i));
    elseif(i ==3)
        mu1 = (2.15-3*0.3)*rand(1,d(i));
    end

    mu2 = zeros(1,d(i));
    cov = eye(d(i));
    rng(37)  % For reproducibility
    design_data = [mvnrnd(mu1,cov,N); mvnrnd(mu2,cov,N)];
    design_label = [ones(N,1); zeros(N,1)];
    test_data = [mvnrnd(mu1,cov,N); mvnrnd(mu2,cov,N)];
    test_label = [ones(N,1); zeros(N,1)];
    g_classifer = fitcnb(design_data,design_label);
    err_true(i) = mean(abs(test_label - round(predict(g_classifer, test_data))));
    for j = 1:length(N_D)
        for l = 1:loop
            ind = [randsample(N, N_D(j)); randsample(N, N_D(j))+N];
            train_data = design_data(ind,:);
            train_label = design_label(ind);
            knn_classifier = fitcknn(train_data,train_label,'NumNeighbors',3);
            g_classifier = fitcnb(train_data,train_label);
            err_test(j,l) = mean(abs(test_label - round(predict(g_classifier, test_data))));
        end
    end
    err_test_mean(i,:) = mean(err_test,2);
end
figure(1)
hold on
for i = 1:length(d)
    plot(N_D,err_test_mean(i,:), 'x-','Linewidth',2,'Markersize',8)
end
xlabel("Size of Training Set")
ylabel("Average Error")
legend("Dimension 5","Dimension 10", "Dimension 15")