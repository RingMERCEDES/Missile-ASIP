% 自动导引导弹追踪目标示例：EKF、UKF 和粒子滤波
% 目标初始位置：(5000, 3000, 2000)
% 导弹初始位置：(4000, 3000, 0)
% 目标做随机变加速曲线运动，导弹做定速运动
% 导弹利用EKF、UKF和PF跟踪并打击目标
% author: R.H.Wang & GPTo1

clear; clc; close all;

%% 参数设置
rng(1); %随机种子
% 时间设置
dt = 0.1;               % 时间步长 (秒)
t_total = 30;           % 总仿真时间 (秒)
N = floor(t_total / dt);% 仿真步数

% 目标初始状态 [x; y; z; vx; vy; vz; ax; ay; az]
X_target = [5000; 3000; 2000; 0; 0; 0; 0; 0; 0];
q_values = [0, 1, 2];    % 不同的过程噪声标准差
% 导弹初始状态 [x; y; z; vx; vy; vz; ax; ay; az]
% 为每个滤波器分别定义导弹状态
X_missile_ekf = [4000; 3000; 0; 0; 0; 0; 0; 0; 0];
X_missile_ukf = X_missile_ekf;
X_missile_pf = X_missile_ekf;

% 状态转移矩阵 (线性)
F = eye(9);
for i = 1:3
    F(i, i+3) = dt;
    F(i, i+6) = 0.5 * dt^2;
    F(i+3, i+6) = dt;
end

% 控制输入矩阵 (无控制输入，此处为零矩阵)
G = zeros(9, 3);  % 假设没有控制输入

% 噪声参数
q = 1;                 % 过程噪声标准差 (加速度)
Q = zeros(9,9);
for i = 1:3
    Q(i, i) = 0.25 * dt^4;
    Q(i, i+3) = 0.5 * dt^3;
    Q(i, i+6) = 0.25 * dt^2;
    Q(i+3, i+3) = dt^2;
    Q(i+3, i+6) = dt;
    Q(i+6, i+6) = 1;
end
Q = Q * q^2;

r = 10;                % 测量噪声标准差 (范围单位米, 角度单位弧度)
R = diag([r^2, (deg2rad(1))^2, (deg2rad(1))^2]);   % 观测噪声协方差

% 粒子滤波参数
num_particles = 1000;

% 导弹速度
V_missile = 300;       % 导弹速度 (m/s)

%% 初始化滤波器

% EKF 初始化
X_ekf = zeros(9, N);
X_ekf(:,1) = X_missile_ekf;  % 初始估计
P_ekf = diag([100^2, 100^2, 100^2, 10^2, 10^2, 10^2, 1^2, 1^2, 1^2]);  % 初始协方差

% UKF 参数
alpha = 1e-3;
kappa = 0;
beta = 2;
n_aug = 9;
lambda = alpha^2 * (n_aug + kappa) - n_aug;
% UKF 权重
Wm = [lambda / (n_aug + lambda), repmat(1/(2*(n_aug + lambda)), 1, 2*n_aug)];
Wc = Wm;
Wc(1) = Wc(1) + (1 - alpha^2 + beta);
% UKF 状态和协方差
X_ukf = zeros(9, N);
X_ukf(:,1) = X_missile_ukf;
P_ukf = P_ekf;

% 粒子滤波初始化
particles = repmat(X_missile_pf, 1, num_particles) + 10 * randn(9, num_particles);  % 初始粒子
weights = ones(1, num_particles) / num_particles;

% 记录误差
error_ekf = zeros(1, N);
error_ukf = zeros(1, N);
error_pf = zeros(1, N);

% 记录轨迹
traj_target = zeros(9, N);
traj_target(:,1) = X_target;
traj_ekf = zeros(9, N);
traj_ekf(:,1) = X_ekf(:,1);
traj_ukf = zeros(9, N);
traj_ukf(:,1) = X_ukf(:,1);
traj_pf = zeros(9, N);
traj_pf(:,1) = X_missile_pf;

% 击中标志
hit_ekf = false;
hit_ukf = false;
hit_pf = false;

% 定义时间向量
time = (0:N-1) * dt;

%% 仿真循环

for k = 2:N
    % 时间步
    t = (k-1) * dt;
    
    %% 目标运动模型
    % 目标加速度为随机
    a_target = q * randn(3,1);
    % 更新目标状态
    traj_target(:,k) = F * traj_target(:,k-1) + [zeros(6,1); a_target];
    
    %% 观测生成
    % 观测噪声
    noise = mvnrnd([0; 0; 0], R)';
    % 计算范围、方位角和俯仰角
    pos_target = traj_target(1:3,k);
    range = norm(pos_target);
    % 避免除以零
    if range == 0
        azimuth = 0;
        elevation = 0;
    else
        azimuth = atan2(pos_target(2), pos_target(1));
        % Clamp z/range to [-1, 1] to avoid NaN in asin
        elevation = asin(max(min(pos_target(3) / range, 1), -1));
    end
    Z = [range; azimuth; elevation] + noise;
    
    %% EKF 预测与更新
    if ~hit_ekf
        % 预测
        X_pred_ekf = F * X_ekf(:,k-1);
        P_pred_ekf = F * P_ekf * F' + Q;
        
        % 观测预测（非线性）
        pos_pred = X_pred_ekf(1:3);
        range_pred = norm(pos_pred);
        if range_pred == 0
            azimuth_pred = 0;
            elevation_pred = 0;
        else
            azimuth_pred = atan2(pos_pred(2), pos_pred(1));
            elevation_pred = asin(max(min(pos_pred(3) / range_pred, 1), -1));
        end
        Z_pred = [range_pred; azimuth_pred; elevation_pred];
        
        % 计算雅可比矩阵 H_jacobian
        if range_pred == 0
            H_jacobian = zeros(3,9);
        else
            H_jacobian = zeros(3,9);
            % ∂range/∂x = x/range, ∂range/∂y = y/range, ∂range/∂z = z/range
            H_jacobian(1,1) = pos_pred(1) / range_pred;
            H_jacobian(1,2) = pos_pred(2) / range_pred;
            H_jacobian(1,3) = pos_pred(3) / range_pred;
            % ∂azimuth/∂x = -y / (x^2 + y^2), ∂azimuth/∂y = x / (x^2 + y^2)
            denom_az = pos_pred(1)^2 + pos_pred(2)^2;
            if denom_az == 0
                H_jacobian(2,1:3) = 0;
            else
                H_jacobian(2,1) = -pos_pred(2) / denom_az;
                H_jacobian(2,2) = pos_pred(1) / denom_az;
                H_jacobian(2,3) = 0;
            end
            % ∂elevation/∂x = -x*z / (range^2 * sqrt(1 - (z/range)^2))
            % ∂elevation/∂y = -y*z / (range^2 * sqrt(1 - (z/range)^2))
            % ∂elevation/∂z = sqrt(x^2 + y^2) / (range^2 * sqrt(1 - (z/range)^2))
            denom_el = range_pred^2 * sqrt(1 - (pos_pred(3)/range_pred)^2);
            if denom_el == 0
                H_jacobian(3,1:9) = 0;
            else
                H_jacobian(3,1) = -pos_pred(1) * pos_pred(3) / denom_el;
                H_jacobian(3,2) = -pos_pred(2) * pos_pred(3) / denom_el;
                H_jacobian(3,3) = sqrt(pos_pred(1)^2 + pos_pred(2)^2) / denom_el;
            end
        end
        
        % 观测残差
        y_ekf = Z - Z_pred;
        
        % 卡尔曼增益
        S_ekf = H_jacobian * P_pred_ekf * H_jacobian' + R;
        K_ekf = P_pred_ekf * H_jacobian' / S_ekf;
        
        % 更新状态和协方差
        X_ekf(:,k) = X_pred_ekf + K_ekf * y_ekf;
        P_ekf = (eye(9) - K_ekf * H_jacobian) * P_pred_ekf;
        
        % 计算误差
        error_ekf(k) = norm(X_ekf(1:3,k) - traj_target(1:3,k));
        
        % 检查是否击中目标
        if error_ekf(k) <= 5
            hit_ekf = true;
            fprintf('EKF击中目标时间: %.2f 秒\n', t);
            T_hit_ekf=t;
        end
    else
        % 保持上一次的估计值
        X_ekf(:,k) = X_ekf(:,k-1);
        error_ekf(k) = error_ekf(k-1);
    end
    
    %% UKF 预测与更新
    if ~hit_ukf
        % 生成sigma点
        [Sigma, Wm_ukf, Wc_ukf] = sigma_points_ukf(X_ukf(:,k-1), P_ukf, alpha, kappa, beta);
        
        % 预测sigma点
        Sigma_pred_ukf = F * Sigma;
        
        % 计算预测均值
        X_pred_ukf = Sigma_pred_ukf * Wm_ukf';
        
        % 计算预测协方差
        P_pred_ukf = Q;
        for i_sigma = 1:size(Sigma_pred_ukf,2)
            y_sigma = Sigma_pred_ukf(:,i_sigma) - X_pred_ukf;
            P_pred_ukf = P_pred_ukf + Wc_ukf(i_sigma) * (y_sigma * y_sigma');
        end
        
        % 预测观测
        Z_sigma_ukf = zeros(3, size(Sigma_pred_ukf,2));
        for i_sigma = 1:size(Sigma_pred_ukf,2)
            pos_sigma = Sigma_pred_ukf(1:3,i_sigma);
            range_sigma = norm(pos_sigma);
            if range_sigma == 0
                azimuth_sigma = 0;
                elevation_sigma = 0;
            else
                azimuth_sigma = atan2(pos_sigma(2), pos_sigma(1));
                % Clamp to avoid NaN
                elevation_sigma = asin(max(min(pos_sigma(3) / range_sigma, 1), -1));
            end
            Z_sigma_ukf(:,i_sigma) = [range_sigma; azimuth_sigma; elevation_sigma];
        end
        Z_pred_ukf = Z_sigma_ukf * Wm_ukf';
        
        % 计算观测协方差和交叉协方差
        P_zz_ukf = R;
        P_xz_ukf = zeros(9,3);
        for i_sigma = 1:size(Sigma_pred_ukf,2)
            dz = Z_sigma_ukf(:,i_sigma) - Z_pred_ukf;
            dx = Sigma_pred_ukf(:,i_sigma) - X_pred_ukf;
            P_zz_ukf = P_zz_ukf + Wc_ukf(i_sigma) * (dz * dz');
            P_xz_ukf = P_xz_ukf + Wc_ukf(i_sigma) * (dx * dz');
        end
        
        % 卡尔曼增益
        K_ukf = P_xz_ukf / P_zz_ukf;
        
        % 更新状态
        y_ukf = Z - Z_pred_ukf;
        X_ukf(:,k) = X_pred_ukf + K_ukf * y_ukf;
        
        % 更新协方差
        P_ukf = P_pred_ukf - K_ukf * P_zz_ukf * K_ukf';
        
        % 计算误差
        error_ukf(k) = norm(X_ukf(1:3,k) - traj_target(1:3,k));
        
        % 检查是否击中目标
        if error_ukf(k) <= 5
            hit_ukf = true;
            fprintf('UKF击中目标时间: %.2f 秒\n', t);
            T_hit_ukf=t;
        end
    else
        % 保持上一次的估计值
        X_ukf(:,k) = X_ukf(:,k-1);
        error_ukf(k) = error_ukf(k-1);
    end
    
    %% 粒子滤波预测与更新
    if ~hit_pf
        % 粒子传播
        particles = F * particles + [zeros(6, num_particles); q * randn(3, num_particles)];
        
        % 计算权重
        weights_pf = zeros(1, num_particles);
        for p = 1:num_particles
            pos_p = particles(1:3, p);
            range_p = norm(pos_p);
            if range_p == 0
                azimuth_p = 0;
                elevation_p = 0;
            else
                azimuth_p = atan2(pos_p(2), pos_p(1));
                % Clamp to avoid NaN
                elevation_p = asin(max(min(pos_p(3) / range_p, 1), -1));
            end
            Z_pred_p = [range_p; azimuth_p; elevation_p];
            % 计算似然，添加小常数防止概率为零
            weights_pf(p) = mvnpdf(Z, Z_pred_p, R) + 1e-300;
        end
        weights_pf = weights_pf / sum(weights_pf);
        
        % 检查是否所有权重为零
        if all(weights_pf == 0)
            warning('所有粒子权重为零，重新初始化粒子');
            particles = repmat(X_missile_pf, 1, num_particles) + 10 * randn(9, num_particles);
            weights_pf = ones(1, num_particles) / num_particles;
        end
        
        % 重采样（系统重采样）
        Neff = 1 / sum(weights_pf.^2);
        if Neff < num_particles / 2
            indices = systematic_resample(weights_pf);
            particles = particles(:, indices);
            weights_pf = ones(1, num_particles) / num_particles;
        end
        
        % 状态估计（加权平均）
        X_pf = particles * weights_pf';
        
        % 计算误差
        error_pf(k) = norm(X_pf(1:3) - traj_target(1:3,k));
        
        % 检查是否击中目标
        if error_pf(k) <= 5
            hit_pf = true;
            fprintf('PF击中目标时间: %.2f 秒\n', t);
            T_hit_pf=t;
        end
    else
        % 保持上一次的估计值
        X_pf = X_pf;
        error_pf(k) = error_pf(k-1);
    end
    
    %% 导弹控制（基于各自滤波器的估计）
    if ~hit_ekf
        % 导弹朝EKF估计的目标位置移动
        direction_ekf = X_ekf(1:3,k) - X_missile_ekf(1:3);
        distance_ekf = norm(direction_ekf);
        if distance_ekf > 0
            direction_ekf = direction_ekf / distance_ekf;
        end
        % 更新导弹速度
        X_missile_ekf(4:6) = V_missile * direction_ekf;
        % 更新导弹位置
        X_missile_ekf(1:3) = X_missile_ekf(1:3) + X_missile_ekf(4:6) * dt;
    end
    
    if ~hit_ukf
        % 导弹朝UKF估计的目标位置移动
        direction_ukf = X_ukf(1:3,k) - X_missile_ukf(1:3);
        distance_ukf = norm(direction_ukf);
        if distance_ukf > 0
            direction_ukf = direction_ukf / distance_ukf;
        end
        % 更新导弹速度
        X_missile_ukf(4:6) = V_missile * direction_ukf;
        % 更新导弹位置
        X_missile_ukf(1:3) = X_missile_ukf(1:3) + X_missile_ukf(4:6) * dt;
    end
    
    if ~hit_pf
        % 导弹朝PF估计的目标位置移动
        direction_pf = X_pf(1:3) - X_missile_pf(1:3);
        distance_pf = norm(direction_pf);
        if distance_pf > 0
            direction_pf = direction_pf / distance_pf;
        end
        % 更新导弹速度
        X_missile_pf(4:6) = V_missile * direction_pf;
        % 更新导弹位置
        X_missile_pf(1:3) = X_missile_pf(1:3) + X_missile_pf(4:6) * dt;
    end
    
    %% 更新轨迹记录
    traj_ekf(1:3,k) = X_missile_ekf(1:3);
    traj_ekf(7:9,k) = X_missile_ekf(4:6);  % 更新导弹速度
    traj_ukf(1:3,k) = X_missile_ukf(1:3);
    traj_ukf(7:9,k) = X_missile_ukf(4:6);
    traj_pf(1:3,k) = X_missile_pf(1:3);
    traj_pf(7:9,k) = X_missile_pf(4:6);
    
    %% 动态动画更新（每隔10步更新一次）
    if mod(k, 10) == 0 || k == N
        figure(1);
        clf;
        hold on; grid on; box on;
        plot3(traj_target(1,1:k), traj_target(2,1:k), traj_target(3,1:k), 'k-', 'LineWidth', 2, 'DisplayName', '目标轨迹');
        plot3(traj_ekf(1,1:k), traj_ekf(2,1:k), traj_ekf(3,1:k), 'r--', 'LineWidth', 2, 'DisplayName', 'EKF 导弹轨迹');
        plot3(traj_ukf(1,1:k), traj_ukf(2,1:k), traj_ukf(3,1:k), 'g--', 'LineWidth', 2, 'DisplayName', 'UKF 导弹轨迹');
        plot3(traj_pf(1,1:k), traj_pf(2,1:k), traj_pf(3,1:k), 'b--', 'LineWidth', 2, 'DisplayName', 'PF 导弹轨迹');
        scatter3(traj_target(1,k), traj_target(2,k), traj_target(3,k), 50, 'k', 'filled', 'DisplayName', '目标当前位置');
        scatter3(traj_ekf(1,k), traj_ekf(2,k), traj_ekf(3,k), 50, 'r', 'filled', 'DisplayName', 'EKF 导弹当前位置');
        scatter3(traj_ukf(1,k), traj_ukf(2,k), traj_ukf(3,k), 50, 'g', 'filled', 'DisplayName', 'UKF 导弹当前位置');
        scatter3(traj_pf(1,k), traj_pf(2,k), traj_pf(3,k), 50, 'b', 'filled', 'DisplayName', 'PF 导弹当前位置');
        xlabel('X (m)');
        ylabel('Y (m)');
        zlabel('Z (m)');
        title(sprintf('目标与滤波器估计轨迹 (时间: %.1f 秒)', t));
        legend('show');
        view(3);
        drawnow;
        pause(0.01); % 控制动画速度
    end
    
    %% 终止仿真如果所有滤波器都击中目标
    if hit_ekf && hit_ukf && hit_pf
        traj_target(:,k+1:N) = traj_target(:,k);
        traj_ekf(:,k+1:N) = traj_ekf(:,k);
        traj_ukf(:,k+1:N) = traj_ukf(:,k);
        traj_pf(:,k+1:N) = traj_pf(:,k);
        error_ekf(k+1:N) = error_ekf(k);
        error_ukf(k+1:N) = error_ukf(k);
        error_pf(k+1:N) = error_pf(k);
        break;
    end
end

%% 结果可视化

% 1. 动态轨迹动画已在仿真循环中完成

% 2. 跟踪位置误差随时间变化
figure;
hold on; grid on; box on;
plot(time, error_ekf, 'r-', 'LineWidth', 2, 'DisplayName', 'EKF');
plot(time, error_ukf, 'g-', 'LineWidth', 2, 'DisplayName', 'UKF');
plot(time, error_pf, 'b-', 'LineWidth', 2, 'DisplayName', 'PF');
ylim([0,3000]);
xlabel('时间 (秒)');
ylabel('位置误差 (米)');
title('跟踪位置误差随时间变化');
legend('show');

% 3. 参数设置对UKF误差的影响（示例：改变过程噪声q）
% q_values = [0.5, 1, 2];    % 不同的过程噪声标准差
error_vs_q = zeros(length(q_values), N);

for i = 1:length(q_values)
    fprintf('正在运行 q = %.2f 的仿真...\n', q_values(i));
    % 重置状态
    X_target_temp = [5000; 3000; 2000; 0; 0; 0; 0; 0; 0];
    X_missile_temp = [4000; 3000; 0; 0; 0; 0; 0; 0; 0];
    % 重定义Q
    q_temp = q_values(i);
    Q_temp = zeros(9,9);
    for j = 1:3
        Q_temp(j, j) = 0.25 * dt^4;
        Q_temp(j, j+3) = 0.5 * dt^3;
        Q_temp(j, j+6) = 0.25 * dt^2;
        Q_temp(j+3, j+3) = dt^2;
        Q_temp(j+3, j+6) = dt;
        Q_temp(j+6, j+6) = 1;
    end
    Q_temp = Q_temp * q_temp^2;
    % 重置UKF
    X_ukf_temp = zeros(9, N);
    X_ukf_temp(:,1) = X_missile_temp;
    P_ukf_temp = diag([100^2, 100^2, 100^2, 10^2, 10^2, 10^2, 1^2, 1^2, 1^2]);
    
    % UKF 权重保持不变
    Wm_ukf_temp = Wm;
    Wc_ukf_temp = Wc;
    
    for k_p = 2:N
        % 时间步
        t_p = (k_p-1) * dt;
        
        %% 目标运动模型
        % 目标加速度为随机
        a_target_p = q_temp * randn(3,1);
        % 更新目标状态
        X_target_temp = F * X_target_temp + [zeros(6,1); a_target_p];
        
        %% 观测生成
        % 观测噪声
        noise_p = mvnrnd([0; 0; 0], R)';
        % 计算范围、方位角和俯仰角
        pos_target_p = X_target_temp(1:3);
        range_p = norm(pos_target_p);
        if range_p == 0
            azimuth_p = 0;
            elevation_p = 0;
        else
            azimuth_p = atan2(pos_target_p(2), pos_target_p(1));
            % Clamp to avoid NaN
            elevation_p = asin(max(min(pos_target_p(3) / range_p, 1), -1));
        end
        Z_p = [range_p; azimuth_p; elevation_p] + noise_p;
        
        %% UKF 预测与更新
        % 生成sigma点
        [Sigma_p, Wm_p, Wc_p] = sigma_points_ukf(X_ukf_temp(:,k_p-1), P_ukf_temp, alpha, kappa, beta);
        
        % 预测sigma点
        Sigma_pred_p = F * Sigma_p;
        
        % 计算预测均值
        X_pred_p = Sigma_pred_p * Wm_p';
        
        % 计算预测协方差
        P_pred_p = Q_temp;
        for i_sigma = 1:size(Sigma_pred_p,2)
            y_sigma = Sigma_pred_p(:,i_sigma) - X_pred_p;
            P_pred_p = P_pred_p + Wc_p(i_sigma) * (y_sigma * y_sigma');
        end
        
        % 预测观测
        Z_sigma_p = zeros(3, size(Sigma_pred_p,2));
        for i_sigma = 1:size(Sigma_pred_p,2)
            pos_sigma_p = Sigma_pred_p(1:3,i_sigma);
            range_sigma_p = norm(pos_sigma_p);
            if range_sigma_p == 0
                azimuth_sigma_p = 0;
                elevation_sigma_p = 0;
            else
                azimuth_sigma_p = atan2(pos_sigma_p(2), pos_sigma_p(1));
                % Clamp to avoid NaN
                elevation_sigma_p = asin(max(min(pos_sigma_p(3) / range_sigma_p, 1), -1));
            end
            Z_sigma_p(:,i_sigma) = [range_sigma_p; azimuth_sigma_p; elevation_sigma_p];
        end
        Z_pred_p = Z_sigma_p * Wm_p';
        
        % 计算观测协方差和交叉协方差
        P_zz_p = R;
        P_xz_p = zeros(9,3);
        for i_sigma = 1:size(Sigma_pred_p,2)
            dz = Z_sigma_p(:,i_sigma) - Z_pred_p;
            dx = Sigma_pred_p(:,i_sigma) - X_pred_p;
            P_zz_p = P_zz_p + Wc_p(i_sigma) * (dz * dz');
            P_xz_p = P_xz_p + Wc_p(i_sigma) * (dx * dz');
        end
        
        % 卡尔曼增益
        K_p = P_xz_p / P_zz_p;
        
        % 更新状态
        y_p = Z_p - Z_pred_p;
        X_ukf_temp(:,k_p) = X_pred_p + K_p * y_p;
        
        % 更新协方差
        P_ukf_temp = P_pred_p - K_p * P_zz_p * K_p';
        
        % 计算误差
        error_ukf_temp = norm(X_ukf_temp(1:3,k_p) - X_target_temp(1:3));
        
        %% 记录UKF误差
        error_vs_q(i,k_p) = error_ukf_temp;
    end
end

% 绘制不同q值下的UKF位置误差
figure;
hold on; grid on; box on;
colors = ['r', 'g', 'b'];
for i = 1:length(q_values)
    plot(time, error_vs_q(i,:), 'Color', colors(i), 'LineWidth', 2, 'DisplayName', ['q = ' num2str(q_values(i))]);
end
ylim([0,3000]);
xlabel('时间 (秒)');
ylabel('UKF位置误差 (米)');
title('不同过程噪声q对UKF位置误差的影响');
legend('show');

%% 辅助函数

% 无迹卡尔曼滤波的sigma点生成函数
function [Sigma, Wm, Wc] = sigma_points_ukf(x, P, alpha, kappa, beta)
    % 生成UKF的sigma点及其权重
    n = length(x);
    lambda = alpha^2 * (n + kappa) - n;
    Sigma = zeros(n, 2*n+1);
    Sigma(:,1) = x;
    sqrt_P = chol((n + lambda) * P)';
    for i = 1:n
        Sigma(:,i+1) = x + sqrt_P(:,i);
        Sigma(:,i+n+1) = x - sqrt_P(:,i);
    end
    Wm = [lambda / (n + lambda), repmat(1/(2*(n + lambda)), 1, 2*n)];
    Wc = Wm;
    Wc(1) = Wc(1) + (1 - alpha^2 + beta);
end

% 系统重采样函数（粒子滤波）
function indices = systematic_resample(weights)
    % 系统重采样算法
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
            if j > N
                j = N;
            end
        end
    end
end
