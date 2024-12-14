% 导弹制导仿真：EKF、UKF 和粒子滤波的对比实验
clear; clc; close all;

%% 仿真参数初始化
delta_t = 0.01;          % 采样周期 (秒)
longa = 10000;           % 机动时间常数的倒数，即机动频率
tf = 3.7;                % 总飞行时间 (秒)
T = ceil(tf / delta_t);  % 采样点数
N_sim = 5;              % 蒙特卡洛仿真次数

%% 系统矩阵定义
F = [eye(3), delta_t * eye(3), (exp(-longa * delta_t) + longa * delta_t - 1) / longa^2 * eye(3); ...
     zeros(3), eye(3), (1 - exp(-longa * delta_t)) / longa * eye(3); ...
     zeros(3), zeros(3), exp(-longa * delta_t) * eye(3)];

G = [-0.5 * delta_t^2 * eye(3); -delta_t * eye(3); zeros(3)];

N = 3;  % 导航增益

%% 噪声参数
cigema = sqrt(0.1);      % 过程噪声标准差
Q = [zeros(6,6), zeros(6,3); zeros(3,6), cigema^2 * eye(3)];  % 过程噪声协方差矩阵
R_factor = 0.1;          % 测量噪声缩放因子

%% 粒子滤波参数
num_particles = 1000;    % 粒子数量

%% 预分配误差矩阵
Ep_ekfx = zeros(N_sim, T-3);
Ep_ekfy = zeros(N_sim, T-3);
Ep_ekfz = zeros(N_sim, T-3);
Ep_ukfx = zeros(N_sim, T-3);
Ep_ukfy = zeros(N_sim, T-3);
Ep_ukfz = zeros(N_sim, T-3);
Ep_pfx = zeros(N_sim, T-3);
Ep_pfy = zeros(N_sim, T-3);
Ep_pfz = zeros(N_sim, T-3);

Ev_ekf = zeros(N_sim, T-3);
Ev_ukf = zeros(N_sim, T-3);
Ev_pff = zeros(N_sim, T-3);

Ea_ekf = zeros(N_sim, T-3);
Ea_ukf = zeros(N_sim, T-3);
Ea_pfa = zeros(N_sim, T-3);

%% 蒙特卡洛仿真主循环
for i = 1:N_sim
    % 初始化真实状态
    x = zeros(9, T);
    x(:,1) = [3500, 1500, 1000, -1100, -150, -50, 0, 0, 0]';
    
    % 初始化 EKF 状态
    ex_ekf = zeros(9, T);
%     ex_ekf(:,1) = [3000, 1200, 960, -800, -100, -100, 0, 0, 0]';
% 可以改变初始状态
    ex_ekf(:,1) = [3500, 1500, 1000, -1100, -150, -50, 0, 0, 0]';
    % 初始化 UKF 状态
    ex_ukf = zeros(9, T);
    ex_ukf(:,1) = ex_ekf(:,1);  % UKF 初始状态与 EKF 相同
    
    % 初始化粒子滤波状态
    particles = repmat(ex_ekf(:,1), 1, num_particles) + 1e2 * randn(9, num_particles);  % 粒子初始分布
    weights = ones(1, num_particles) / num_particles;  % 粒子权重初始化
    
    % 初始化协方差矩阵
    P0_ekf = [10^4 * eye(6), zeros(6,3); zeros(3,6), 10^2 * eye(3)];
    P0_ukf = P0_ekf;
    
    % 初始化测量
    z = zeros(2, T);
    z(:,1) = [atan(x(2,1) / sqrt(x(1,1)^2 + x(3,1)^2)); atan(-x(3,1) / x(1,1))];
    
    % 生成过程噪声
    w = [zeros(6, T); cigema * randn(3, T)];
    
    % 初始化测量噪声
    d = sqrt(x(1,1)^2 + x(2,1)^2 + x(3,1)^2);
    D = [d, 0; 0, d];
    R_init = inv(D) * R_factor * eye(2) * inv(D)';
    v = zeros(2, T);
    
    % 生成真实轨迹和测量值
    for k = 2:T-3
        tgo = tf - k * delta_t + 1e-16;  % 剩余时间
        c1 = N / tgo^2;
        c2 = N / tgo;
        c3 = N * (exp(-longa * tgo) + longa * tgo - 1) / (longa * tgo)^2;
        
        % 计算控制输入
        u = [c1, c2, c3] * [x(1,k-1); x(4,k-1); x(7,k-1)];
        u = [u; [c1, c2, c3] * [x(2,k-1); x(5,k-1); x(8,k-1)];
             [c1, c2, c3] * [x(3,k-1); x(6,k-1); x(9,k-1)]];
        
        % 更新真实状态
        x(:,k) = F * x(:,k-1) + G * u + w(:,k-1);
        
        % 生成测量值
        d = sqrt(x(1,k)^2 + x(2,k)^2 + x(3,k)^2);
        D = [d, 0; 0, d];
        R = inv(D) * R_factor * eye(2) * inv(D)';
        v(:,k) = sqrtm(R) * randn(2,1);
        z(:,k) = [atan(x(2,k) / sqrt(x(1,k)^2 + x(3,k)^2)); ...
                  atan(-x(3,k) / x(1,k))] + v(:,k);  % 使用分号确保为 2x1 列向量
    end
    
    %% EKF 初始化
    eP_ekf = P0_ekf;
    
    %% UKF 初始化
    alpha = 1e-3;
    kappa = 0;
    beta = 2;
    [Sigma_ukf, Wm, Wc] = sigma_points(ex_ukf(:,1), P0_ukf, alpha, kappa, beta);
    
    %% 粒子滤波初始化
    % 粒子已经在循环开始时初始化
    
    %% 滤波主循环
    for k = 2:T-3
        % 当前时间步
        tgo = tf - k * delta_t + 1e-16;
        c1 = N / tgo^2;
        c2 = N / tgo;
        c3 = N * (exp(-longa * tgo) + longa * tgo - 1) / (longa * tgo)^2;
        u_control = [c1, c2, c3];
        
        %% EKF 预测与更新
        % 计算控制输入
        u_ekf = u_control * [ex_ekf(1,k-1); ex_ekf(4,k-1); ex_ekf(7,k-1)];
        u_ekf = [u_ekf; u_control * [ex_ekf(2,k-1); ex_ekf(5,k-1); ex_ekf(8,k-1)];
                 u_control * [ex_ekf(3,k-1); ex_ekf(6,k-1); ex_ekf(9,k-1)]];
        
        % EKF 预测步骤
        Xn_ekf = F * ex_ekf(:,k-1) + G * u_ekf;
        P_pred_ekf = F * eP_ekf * F' + Q;
        
        % EKF 测量预测
        Zn_ekf = [atan(Xn_ekf(2) / sqrt(Xn_ekf(1)^2 + Xn_ekf(3)^2)); ...
                  atan(-Xn_ekf(3) / Xn_ekf(1))];
        
        % 计算雅可比矩阵 H
        H = compute_jacobian(Xn_ekf);
        
        % EKF 卡尔曼增益
        S = H * P_pred_ekf * H' + R_factor * eye(2);  % 假设 R = R_factor * I
        K_ekf = P_pred_ekf * H' / S;
        
        % EKF 更新步骤
        ex_ekf(:,k) = Xn_ekf + K_ekf * (z(:,k) - Zn_ekf);
        eP_ekf = (eye(9) - K_ekf * H) * P_pred_ekf;
        
        %% UKF 预测与更新
        % 生成 sigma 点
        [Sigma_ukf, Wm, Wc] = sigma_points(ex_ukf(:,k-1), P0_ukf, alpha, kappa, beta);
        
        % 传播 sigma 点通过系统动态
        % 修正这里的控制输入乘法，确保维度匹配
        % 计算每个 sigma 点的控制输入
        % Sigma_ukf(1,:) -> x
        % Sigma_ukf(4,:) -> y
        % Sigma_ukf(7,:) -> z
        % u_control 是 [c1 c2 c3]，需要与 [x; y; z] 进行矩阵乘法，得到 1x19
        U_control = u_control * [Sigma_ukf(1,:); Sigma_ukf(4,:); Sigma_ukf(7,:)] ;  % 1x19
        
        % 扩展为 3x19 以匹配 G 的维度 (G 是 9x3)
        U_control_repmat = repmat(U_control, 3, 1);  % 3x19
        
        % 计算 Sigma_pred
        Sigma_pred = F * Sigma_ukf + G * U_control_repmat;  % (9x9)*(9x19) + (9x3)*(3x19) = 9x19
        
        % 添加过程噪声
        % 注意，过程噪声应该已在真实状态生成部分添加，不需要在预测中再次添加
        % 因此，这里可以省略 w(:,k-1) * ones(1, num_particles)
        % 或根据实际需求决定是否添加
        % Sigma_pred = Sigma_pred + w(:,k-1) * ones(1, size(Sigma_pred,2));
        % 这里暂时不添加
        
        % 预测均值和协方差
        x_pred = Sigma_pred * Wm';
        P_pred_ukf = Q;
        for p = 1:size(Sigma_pred,2)
            y = Sigma_pred(:,p) - x_pred;
            P_pred_ukf = P_pred_ukf + Wc(p) * (y * y');
        end
        
        % 测量预测
        Z_sigma = zeros(2, size(Sigma_pred,2));
        for p = 1:size(Sigma_pred,2)
            Z_sigma(:,p) = [atan(Sigma_pred(2,p) / sqrt(Sigma_pred(1,p)^2 + Sigma_pred(3,p)^2)); ...
                           atan(-Sigma_pred(3,p) / Sigma_pred(1,p))];
        end
        z_pred = Z_sigma * Wm';
        
        % 测量协方差和交叉协方差
        P_zz = R_factor * eye(2);
        P_xz = zeros(9,2);
        for p = 1:size(Sigma_pred,2)
            dz = Z_sigma(:,p) - z_pred;
            dx = Sigma_pred(:,p) - x_pred;
            P_zz = P_zz + Wc(p) * (dz * dz');
            P_xz = P_xz + Wc(p) * (dx * dz');
        end
        
        % UKF 卡尔曼增益
        K_ukf = P_xz / P_zz;
        
        % UKF 更新步骤
        ex_ukf(:,k) = x_pred + K_ukf * (z(:,k) - z_pred);
        
        % UKF 更新协方差
        P0_ukf = P_pred_ukf - K_ukf * P_zz * K_ukf';
        
        %% 粒子滤波预测与更新
        % 粒子传播
        % 计算每个粒子的控制输入
        U_particles = u_control * [particles(1,:); particles(4,:); particles(7,:)] ;  % 1xN
        U_particles_repmat = repmat(U_particles, 3, 1);  % 3xN
        particles = F * particles + G * U_particles_repmat;  % (9x9)*(9xN) + (9x3)*(3xN) = 9xN
        
        % 计算权重（基于测量似然）
        z_pred_particles = [atan(particles(2,:) ./ sqrt(particles(1,:).^2 + particles(3,:).^2)); ...
                            atan(-particles(3,:) ./ particles(1,:))];
        z_pred_particles = real(z_pred_particles);  % 避免复数
        measurement_diff = z(:,k) - z_pred_particles;  % 2xN
        exponent = -0.5 * sum(measurement_diff.^2, 1) / R_factor;  % 1xN
        weights = exp(exponent);
        weights = weights / sum(weights);  % 归一化权重
        
        % 重采样（系统重采样）
        if 1 / sum(weights.^2) < num_particles / 2
            indices = systematic_resample(weights);
            particles = particles(:, indices);
            weights = ones(1, num_particles) / num_particles;
        end
        
        % 状态估计（粒子均值）
        ex_p = mean(particles, 2);
        
        % 存储粒子滤波估计
        if i ==1 && k ==2
            ex_pf = zeros(9, T);
            ex_pf(:,1) = ex_p;
        end
        ex_pf(:,k) = ex_p;
        
        %% 误差计算
        % 位置误差
        Ep_ekfx(i,k-1) = abs(ex_ekf(1,k) - x(1,k));
        Ep_ekfy(i,k-1) = abs(ex_ekf(2,k) - x(2,k));
        Ep_ekfz(i,k-1) = abs(ex_ekf(3,k) - x(3,k));
        
        Ep_ukfx(i,k-1) = abs(ex_ukf(1,k) - x(1,k));
        Ep_ukfy(i,k-1) = abs(ex_ukf(2,k) - x(2,k));
        Ep_ukfz(i,k-1) = abs(ex_ukf(3,k) - x(3,k));
        
        Ep_pfx(i,k-1) = abs(ex_p(1) - x(1,k));
        Ep_pfy(i,k-1) = abs(ex_p(2) - x(2,k));
        Ep_pfz(i,k-1) = abs(ex_p(3) - x(3,k));
        
        % 速度误差
        Ev_ekf(i,k-1) = sqrt((ex_ekf(4,k) - x(4,k))^2 + ...
                             (ex_ekf(5,k) - x(5,k))^2 + ...
                             (ex_ekf(6,k) - x(6,k))^2);
                         
        Ev_ukf(i,k-1) = sqrt((ex_ukf(4,k) - x(4,k))^2 + ...
                             (ex_ukf(5,k) - x(5,k))^2 + ...
                             (ex_ukf(6,k) - x(6,k))^2);
                         
        Ev_pff(i,k-1) = sqrt((ex_p(4) - x(4,k))^2 + ...
                             (ex_p(5) - x(5,k))^2 + ...
                             (ex_p(6) - x(6,k))^2);
                         
        % 加速度误差
        Ea_ekf(i,k-1) = sqrt((ex_ekf(7,k) - x(7,k))^2 + ...
                             (ex_ekf(8,k) - x(8,k))^2 + ...
                             (ex_ekf(9,k) - x(9,k))^2);
                         
        Ea_ukf(i,k-1) = sqrt((ex_ukf(7,k) - x(7,k))^2 + ...
                             (ex_ukf(8,k) - x(8,k))^2 + ...
                             (ex_ukf(9,k) - x(9,k))^2);
                         
        Ea_pfa(i,k-1) = sqrt((ex_p(7) - x(7,k))^2 + ...
                             (ex_p(8) - x(8,k))^2 + ...
                             (ex_p(9) - x(9,k))^2);
    end
end

%% 计算 RMS 误差
error_r_ekf = mean(Ep_ekfx, 1);
error_r_ukf = mean(Ep_ukfx, 1);
error_r_pf = mean(Ep_pfx, 1);

error_v_ekf = mean(Ev_ekf, 1);
error_v_ukf = mean(Ev_ukf, 1);
error_v_pf = mean(Ev_pff, 1);

error_a_ekf = mean(Ea_ekf, 1);
error_a_ukf = mean(Ea_ukf, 1);
error_a_pf = mean(Ea_pfa, 1);

%% 时间向量
t_vec = delta_t:delta_t:(T-3)*delta_t;

%% 可视化
figure
hold on; box on; grid on;
% 设置图像视角
view(3);
xlabel('x/m');
ylabel('y/m');
zlabel('z/m');
title('跟踪轨迹');
legend('真实轨迹', 'EKF 轨迹', 'UKF 轨迹', 'PF 轨迹', 'Location', 'Best');

% 初始化图例
% 绘制轨迹
user_plot = plot3(x(1,1), x(2,1), x(3,1), '-k.', 'MarkerSize', 10, 'DisplayName', '敌机轨迹');
ekf_plot = plot3(ex_ekf(1,1), ex_ekf(2,1), ex_ekf(3,1), '-r*', 'MarkerFace', 'r', 'DisplayName', 'EKF 轨迹');
ukf_plot = plot3(ex_ukf(1,1), ex_ukf(2,1), ex_ukf(3,1), '-g*', 'MarkerFace', 'g', 'DisplayName', 'UKF 轨迹');
pf_plot = plot3(ex_pf(1,1), ex_pf(2,1), ex_pf(3,1), '-b*', 'MarkerFace', 'b', 'DisplayName', 'PF 轨迹');
% 更新轨迹
for k = 1:T-3
    % 更新用户轨迹
    set(user_plot, 'XData', x(1, 1:k), 'YData', x(2, 1:k), 'ZData', x(3, 1:k));
    % 更新导弹轨迹
    set(ekf_plot, 'XData', ex_ekf(1, 1:k), 'YData', ex_ekf(2, 1:k), 'ZData', ex_ekf(3, 1:k));
    set(ukf_plot, 'XData', ex_ukf(1, 1:k), 'YData', ex_ukf(2, 1:k), 'ZData', ex_ukf(3, 1:k));
    set(pf_plot, 'XData', ex_pf(1, 1:k), 'YData', ex_pf(2, 1:k), 'ZData', ex_pf(3, 1:k));
    % 动画效果：每次更新暂停0.05秒，使运动更平滑
    pause(0.05);
end



% 误差图：位置误差
figure
hold on; box on; grid on;
plot(t_vec, error_r_ekf, '-r', 'DisplayName', 'EKF');
plot(t_vec, error_r_ukf, '-g', 'DisplayName', 'UKF');
plot(t_vec, error_r_pf, '-b', 'DisplayName', 'PF');
xlabel('飞行时间 (秒)');
ylabel('位置估计误差 (米)');
title('位置估计误差随时间变化');
legend('show');

% 误差图：速度误差
figure
hold on; box on; grid on;
plot(t_vec, error_v_ekf, '-r', 'DisplayName', 'EKF');
plot(t_vec, error_v_ukf, '-g', 'DisplayName', 'UKF');
plot(t_vec, error_v_pf, '-b', 'DisplayName', 'PF');
xlabel('飞行时间 (秒)');
ylabel('速度估计误差 (m/s)');
title('速度估计误差随时间变化');
legend('show');

% 误差图：加速度误差
figure
hold on; box on; grid on;
plot(t_vec, error_a_ekf, '-r', 'DisplayName', 'EKF');
plot(t_vec, error_a_ukf, '-g', 'DisplayName', 'UKF');
plot(t_vec, error_a_pf, '-b', 'DisplayName', 'PF');
xlabel('飞行时间 (秒)');
ylabel('加速度估计误差 (m/s^2)');
title('加速度估计误差随时间变化');
legend('show');

%% 辅助函数

% 计算 EKF 的雅可比矩阵 H
function H = compute_jacobian(X)
    % 提取状态变量
    x = X(1);
    y = X(2);
    z = X(3);
    d_sq = x^2 + y^2 + z^2;
    denom1 = sqrt(x^2 + z^2);
    denom2 = x^2 + z^2;
    
    % 计算偏导数
    dh1_dx = -x * y / (d_sq * denom1);
    dh1_dy = denom1 / d_sq;
    dh1_dz = -y * z / (d_sq * denom1);
    
    dh2_dx = z / denom2;
    dh2_dy = 0;
    dh2_dz = -x / denom2;
    
    % 构建雅可比矩阵 H
    H = zeros(2,9);
    H(1,1) = dh1_dx;
    H(1,2) = dh1_dy;
    H(1,3) = dh1_dz;
    H(2,1) = dh2_dx;
    H(2,3) = dh2_dz;
end

% 生成 UKF 的 sigma 点
function [Sigma, Wm, Wc] = sigma_points(x, P, alpha, kappa, beta)
    % 计算参数
    n = length(x);
    lambda = alpha^2 * (n + kappa) - n;
    Sigma = zeros(n, 2*n+1);
    Sigma(:,1) = x;
    
    % 计算平方根矩阵
    sqrt_P = chol((n + lambda) * P)';
    for i = 1:n
        Sigma(:,i+1) = x + sqrt_P(:,i);
        Sigma(:,i+n+1) = x - sqrt_P(:,i);
    end
    
    % 计算权重
    Wm = [lambda / (n + lambda), repmat(1/(2*(n + lambda)), 1, 2*n)];
    Wc = Wm;
    Wc(1) = Wc(1) + (1 - alpha^2 + beta);
end

% 系统重采样算法
function indices = systematic_resample(weights)
    N = length(weights);
    positions = (rand + (0:N-1)) / N;
    cumulative_sum = cumsum(weights);
    indices = zeros(1, N);
    i = 1;
    j = 1;
    while i <= N
        if positions(i) < cumulative_sum(j)
            indices(i) = j;
            i = i + 1;
        else
            j = j + 1;
        end
    end
end
