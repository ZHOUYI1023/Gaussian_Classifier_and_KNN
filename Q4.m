mu1 = [3, 2];
mu2 = [6, 5];
cov1 = [4, 2; 2, 4];
cov2 = [4, 2; 2, 4];
d = [0,0;1,1;2,2;3,3;4,4;5,5;6,6;7,7;8,8];
N = 500;
loop = 10;
rng('default')  % For reproducibility
d_mahal = zeros(length(d),1);
err_design = zeros(length(d),loop);
err_test = zeros(length(d),loop);
for i = 1:length(d)
design_data = [mvnrnd(mu1,cov1,N); mvnrnd(mu2+d(i,:),cov2,N)];
design_label = [ones(N,1); zeros(N,1)];
test_data = [mvnrnd(mu1,cov1,N); mvnrnd(mu2+d(i,:),cov2,N)];
test_label = [ones(N,1); zeros(N,1)];
d_mahal(i) = mahal(mu2+d(i,:),design_data(1:N,:))
(mu2+d(i,:)-mu1)*inv(cov1)*(mu2+d(i,:)-mu1)'
for j = 1:loop
  g_classifer = fitcnb(design_data,design_label);
  err_design(i,j) = mean(abs(design_label - round(predict(g_classifer, design_data))));
  err_test(i,j) = mean(abs(test_label - round(predict(g_classifer, test_data))));
end
end
err_design_mean = mean(err_design,2);
err_test_mean = mean(err_test,2);
figure(1)
hold on
plot(d_mahal,err_design_mean, 'rx-','Linewidth',2,'Markersize',8)
plot(d_mahal,err_test_mean, 'bx-','Linewidth',2,'Markersize',8)
xlabel("Mahalonobis Distance")
ylabel("Average Error")
legend("Mean of E_{train}","Mean of E_{test}")