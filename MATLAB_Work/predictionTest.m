close all; clear; clc
% =========================================================================
%  TLE -> state vectors -> batch least-squares orbit fit -> prediction
%  -> comparison with the "actual" (TLE/SGP4-derived) states
%
%  Method: batch weighted least squares (Gauss-Newton differential
%  correction). The unknowns are the satellite state x0 = [r0; v0] at the
%  start of the fit window (plus, optionally, a constant along-track
%  acceleration aT that absorbs drag/low thrust). x0 is propagated with a
%  numerical force model (two-body + J2, J3, J4) and adjusted until the
%  propagated r,v best match all the TLE-derived r,v in the fit window.
%  The fitted state is then propagated past the fit window and compared
%  with the TLE states that were NOT used in the fit.
% =========================================================================

%% ------------------------------ Settings --------------------------------
tleFile  = "41179_03-01-26_04-01-26.txt";

fitStart = datetime(2026,3,1, 0,0,0,'TimeZone','UTC');   % fit window start
fitEnd   = datetime(2026,3,15, 0,0,0,'TimeZone','UTC');  % fit window end
predEnd  = datetime(2026,3,31,23,0,0,'TimeZone','UTC');  % predict until

estAccel = true;    % also estimate constant along-track acceleration aT
sigR     = 300;     % assumed 1-sigma TLE position error  [m]
sigV     = 0.3;     % assumed 1-sigma TLE velocity error  [m/s]
maxIter  = 15;      % Gauss-Newton iterations
dtDense  = 60;      % step for the dense predicted trajectory [s]

%% -------------- 1. Read TLEs -> r, v (GCRF, m and m/s) ------------------
lines = readlines(tleFile);
i1 = find(startsWith(lines,"1 "));
N = numel(i1);
temp = [tempname '.tle'];
t = NaT(N,1); t.TimeZone = "UTC";
r = zeros(3,N); v = zeros(3,N);

for k = 1:N
    writelines(lines(i1(k):i1(k)+1),temp);
    tle = tleread(temp);
    t(k) = tle.Epoch;
    [r(:,k), v(:,k)] = propagateOrbit(t(k),tle);
end
delete(temp);

[t, idx] = sort(t);
r = r(:,idx); v = v(:,idx);

%% -------------- 2. Remove duplicate TLEs (same epoch +/- 1 s) ------------
[~, iu] = uniquetol(seconds(t - t(1)), 1, 'DataScale', 1);
iu = sort(iu);
fprintf('%d TLEs read, %d unique epochs kept\n', N, numel(iu));
t = t(iu); r = r(:,iu); v = v(:,iu);

%% -------------- 3. Constants and Earth-pole frame -----------------------
C.mu = 3.986004418e14;      % m^3/s^2
C.Re = 6378137.0;           % m
C.J2 = 1.08262668e-3;
C.J3 = -2.53265649e-6;
C.J4 = -1.61962159e-6;

% propagateOrbit returns GCRF (J2000-like) vectors, but the Earth's
% oblateness is symmetric about the CURRENT spin axis, which in 2026 is
% tilted ~0.35 deg from GCRF z (precession). Using GCRF z as the J2 axis
% would cause km/day errors, so all fitting is done in a frame whose
% z-axis is the Earth's true pole.
dcm = dcmeci2ecef('IAU-2000/2006', datevec(t(1)));
pole = dcm(3,:).';                          % ITRF z-axis expressed in GCRF
e1 = cross([0;1;0], pole); e1 = e1/norm(e1);
e2 = cross(pole, e1);
R  = [e1.'; e2.'; pole.'];                  % GCRF -> pole frame
rP = R*r;  vP = R*v;

%% -------------- 4. Select fit and comparison data -----------------------
iFit  = find(t >= fitStart & t <= fitEnd);
iComp = find(t >= fitStart & t <= predEnd);
if numel(iFit) < 3
    error('Need at least 3 TLE epochs inside the fit window.');
end
tRef = t(iFit(1));                           % epoch of estimated state
tFit = seconds(t(iFit) - tRef);
m    = numel(iFit);

%% -------------- 5. Batch weighted least squares (Gauss-Newton) ----------
nP = 6 + estAccel;
p  = [rP(:,iFit(1)); vP(:,iFit(1)); zeros(estAccel,1)];  % initial guess
y  = reshape([rP(:,iFit); vP(:,iFit)], [], 1);           % observations
sw = repmat([ones(3,1)/sigR; ones(3,1)/sigV], m, 1);     % sqrt(weights)
step = [10 10 10 0.01 0.01 0.01 1e-8].';                 % finite-diff steps
step = step(1:nP);

fprintf('\nIter   |dr0| [m]    |dv0| [m/s]   wRMS\n');
for it = 1:maxIter
    h   = stateModel(p, tFit, C);
    res = y - h;

    % Jacobian d(h)/d(p) by central differences
    H = zeros(6*m, nP);
    for j = 1:nP
        e = zeros(nP,1); e(j) = step(j);
        H(:,j) = (stateModel(p+e,tFit,C) - stateModel(p-e,tFit,C)) / (2*step(j));
    end

    dx = (sw.*H) \ (sw.*res);     % weighted normal-equation solution
    p  = p + dx;
    fprintf('%3d  %11.3f  %12.6f  %8.4f\n', it, norm(dx(1:3)), ...
            norm(dx(4:6)), sqrt(mean((sw.*res).^2)));
    % Stop when the weighted RMS stops improving (steps of ~1 m are at the
    % integrator / finite-difference noise floor and will never reach 0)
    wrms = sqrt(mean((sw.*res).^2));
    if it > 1 && abs(wrmsPrev - wrms) < 1e-5*wrms, break; end
    wrmsPrev = wrms;
end

% Final residuals and formal covariance (scaled by a-posteriori variance)
res  = y - stateModel(p, tFit, C);
s2   = sum((sw.*res).^2) / (6*m - nP);
% Columns of H have very different units (m, m/s, m/s^2), so scale them
% before inverting to avoid an ill-conditioned normal matrix
A    = sw.*H;
D    = diag(1 ./ vecnorm(A));
As   = A*D;
Pcov = s2 * D * ((As.'*As) \ eye(nP)) * D;
sig  = sqrt(diag(Pcov));

fprintf('\nEstimated state at %s UTC (GCRF):\n', string(tRef));
fprintf('  r0 = [%14.3f %14.3f %14.3f] m\n', R.'*p(1:3));
fprintf('  v0 = [%14.6f %14.6f %14.6f] m/s\n', R.'*p(4:6));
fprintf('  1-sigma r0: %.1f %.1f %.1f m,  v0: %.4f %.4f %.4f m/s\n', sig(1:6));
if estAccel
    fprintf('  along-track accel aT = %.3e +/- %.1e m/s^2\n', p(7), sig(7));
end

%% -------------- 6. Predict and compare with actual states ---------------
tCmp = seconds(t(iComp) - tRef);
Xhat = propagateState(p, tCmp, C);
rHat = R.'*Xhat(1:3,:);   vHat = R.'*Xhat(4:6,:);      % back to GCRF
rAct = r(:,iComp);        vAct = v(:,iComp);
tC   = t(iComp);
isFit = tC <= fitEnd;

dr = rHat - rAct;                    % position error  [m]
dv = vHat - vAct;                    % velocity error  [m/s]
drMag = vecnorm(dr);  dvMag = vecnorm(dv);

% Radial / In-track / Cross-track components of the position error
ric = zeros(3, numel(iComp));
for k = 1:numel(iComp)
    uR = rAct(:,k)/norm(rAct(:,k));
    uC = cross(rAct(:,k), vAct(:,k)); uC = uC/norm(uC);
    uI = cross(uC, uR);
    ric(:,k) = [uR uI uC].' * dr(:,k);
end

fprintf('\nFit arc (%d epochs): RMS |dr| = %.3f km, RMS |dv| = %.3f m/s\n', ...
        nnz(isFit), sqrt(mean(drMag(isFit).^2))/1e3, sqrt(mean(dvMag(isFit).^2)));
fprintf('\nPrediction vs actual (epochs not used in the fit):\n');
fprintf('  %-20s %8s %9s %9s %9s %9s %9s\n', 'Epoch (UTC)', 'dt [d]', ...
        '|dr| km', 'R km', 'I km', 'C km', '|dv| m/s');
for k = find(~isFit).'          % row vector, so the loop runs once per epoch
    fprintf('  %-20s %8.3f %9.3f %9.3f %9.3f %9.3f %9.3f\n', ...
        string(tC(k),'yyyy-MM-dd HH:mm:ss'), days(tC(k)-fitEnd), ...
        drMag(k)/1e3, ric(:,k)/1e3, dvMag(k));
end

% Dense predicted trajectory (every dtDense seconds)
tDense = (0:dtDense:seconds(predEnd - tRef)).';
Xd = propagateState(p, tDense, C);
rDense = R.'*Xd(1:3,:);
tDenseUTC = tRef + seconds(tDense);

%% -------------- 7. Plots -------------------------------------------------
% 3-D trajectory
figure('Name','Trajectory');
[xs,ys,zs] = sphere(60);
surf(C.Re*xs/1e3, C.Re*ys/1e3, C.Re*zs/1e3, 'EdgeColor','none', ...
     'FaceColor',[0.3 0.5 0.9], 'FaceAlpha',0.4); hold on
plot3(rDense(1,:)/1e3, rDense(2,:)/1e3, rDense(3,:)/1e3, 'r-', 'LineWidth',0.5);
plot3(rAct(1,isFit)/1e3,  rAct(2,isFit)/1e3,  rAct(3,isFit)/1e3,  'ko', 'MarkerFaceColor','k');
plot3(rAct(1,~isFit)/1e3, rAct(2,~isFit)/1e3, rAct(3,~isFit)/1e3, 'gs', 'MarkerFaceColor','g');
axis equal; grid on; xlabel('X [km]'); ylabel('Y [km]'); zlabel('Z [km]');
legend('Earth','Predicted trajectory','TLE states used in fit', ...
       'TLE states (actual) in prediction','Location','best');
title('Least-squares fitted and predicted trajectory (GCRF)');

% Position error magnitude
figure('Name','Errors');
subplot(3,1,1)
plot(tC, drMag/1e3, 'o-', 'LineWidth',1.2); grid on
xline(fitEnd, '--', 'End of fit'); ylabel('|\Deltar| [km]');
title('Predicted minus actual');

% RIC components
subplot(3,1,2)
plot(tC, ric/1e3, 'o-', 'LineWidth',1.2); grid on
xline(fitEnd, '--'); ylabel('Position error [km]');
legend('Radial','In-track','Cross-track','Location','best');

% Velocity error magnitude
subplot(3,1,3)
plot(tC, dvMag, 'o-', 'LineWidth',1.2); grid on
xline(fitEnd, '--'); ylabel('|\Deltav| [m/s]'); xlabel('Epoch (UTC)');

% Component-by-component position comparison
figure('Name','Position components');
lbl = {'X','Y','Z'};
for c = 1:3
    subplot(3,1,c)
    plot(tDenseUTC, rDense(c,:)/1e3, 'r-'); hold on
    plot(tC, rAct(c,:)/1e3, 'ko', 'MarkerFaceColor','k');
    xline(fitEnd, '--'); grid on; ylabel([lbl{c} ' [km]']);
    if c == 1, legend('Predicted','Actual (TLE)','Location','best'); end
end
xlabel('Epoch (UTC)');

%% ======================== Local functions ===============================
function h = stateModel(p, tOut, C)
% Stacked [r; v] at each output time, as one column (6*m x 1)
    X = propagateState(p, tOut, C);
    h = X(:);
end

function X = propagateState(p, tOut, C)
% Propagate p = [r0; v0; (aT)] from t = 0 to the times tOut [s] (>= 0)
    tOut  = tOut(:);
    tspan = unique([0; tOut]);
    if numel(tspan) == 2                       % ode113 needs >2 points to
        tspan = [tspan(1); mean(tspan); tspan(2)]; % return only these times
    end
    aT = 0;
    if numel(p) > 6, aT = p(7); end
    opts = odeset('RelTol',1e-11, 'AbsTol',1e-6);
    [~, Y] = ode113(@(tt,x) eom(x, aT, C), tspan, p(1:6), opts);
    [~, loc] = ismember(tOut, tspan);
    X = Y(loc,:).';
end

function dx = eom(x, aT, C)
% Equations of motion: zonal gravity + constant along-track acceleration
    r = x(1:3); v = x(4:6);
    dx = [v; zonalAccel(r, C) + aT * v / norm(v)];
end

function a = zonalAccel(r, C)
% Two-body + J2 + J3 + J4 acceleration, z-axis along Earth's pole
    x = r(1); y = r(2); z = r(3);
    rn = norm(r); z2 = (z/rn)^2;
    a0 = -C.mu * r / rn^3;

    k2 = -1.5 * C.J2 * C.mu * C.Re^2 / rn^5;
    a2 = k2 * [x*(1-5*z2); y*(1-5*z2); z*(3-5*z2)];

    k3 = -2.5 * C.J3 * C.mu * C.Re^3 / rn^7;
    f3 = 3*z - 7*z^3/rn^2;
    a3 = k3 * [x*f3; y*f3; 6*z^2 - 7*z^4/rn^2 - 0.6*rn^2];

    k4 = 15/8 * C.J4 * C.mu * C.Re^4 / rn^7;
    f4 = 1 - 14*z2 + 21*z2^2;
    a4 = k4 * [x*f4; y*f4; z*(5 - 70/3*z2 + 21*z2^2)];

    a = a0 + a2 + a3 + a4;
end