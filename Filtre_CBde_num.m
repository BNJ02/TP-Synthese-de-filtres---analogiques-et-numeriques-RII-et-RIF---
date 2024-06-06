clc
clear all

% Fréquences d'échantillonnage et de tracé
duree = 1;
fe = 8192;   % Fréquence d'échantillonnage (Hz)
fmin = 250;  % Fréquence minimale de tracé (Hz)
fmax = 550;  % Fréquence maximale de tracé (Hz)
samples = 1000;    % Nombre de points de tracé

% Retourne le signal en fonction du temps
[sig,t] = gamme(duree, fe);

% Produit le son du signal avec la fe sélectionnée
% soundsc(sig, fe)

% Transformée de Fourier de notre signal + affichage spectre
S = fft(sig)/fe;
f = [0:length(sig)-1]*(fe/length(S));
fig = figure;
set(fig, 'Name', 'Spectre signal', 'NumberTitle', 'off');
plot(f, abs(S)); 
axis([fmin fmax 0 0.6])

% Spécifications du filtre
Ap = 1;     % Atténuation maximale en bande passante (dB)
Aa = 40;    % Atténuation minimale en bande atténuée (dB)
fbc = 340;   % Fréquence de coupure basse (Hz)
fhc = 360;   % Fréquence de coupure haute (Hz)
df = 10;   % Largeur de la bande de transition (Hz)

% Définition des fréquences atténuation et passante
fbp = fbc - df/2;
fba = fbc + df/2;
fhp = fhc + df/2;
fha = fhc - df/2;

fpass = [fbp fhp];
fstop = [fba fha];

% Normalisation
Fp = fpass / (fe/2);
Fs = fstop / (fe/2);

% Calcul de l'ordre et des coefficients du filtre de Chebyshev de type 2
[n,wn] = cheb2ord(Fp,Fs,Ap,Aa);
[Bc2,Ac2] = cheby2(n,Aa,wn,'stop');

% Affichage de l'ordre du filtre
disp(['Ordre du filtre de Chebyshev de type 2 : ', num2str(n)]);

f = linspace(fmin,fmax,samples);

% Calcul de la réponse fréquentielle en amplitude
Hc1 = freqz(Bc2, Ac2, f, fe);
fig = figure;
set(fig, 'Name', 'Réponse freq filtre RII', 'NumberTitle', 'off');
plot(f, 20*log10(abs(Hc1)));
xlabel('Fréquence (Hz)');
ylabel('Amplitude (dB)');
legend('Chebyshev 2');
grid on;

% Application du filtre au signal audio synthétique
y = filter(Bc2,Ac2,sig);

% Ecoute du signal après filtrage
%sound(y,fe);

%%% Chebychev2 %%%
fig = figure;
set(fig, 'Name', 'Signal filtré & spectre amplitude RII', 'NumberTitle', 'off');
% Tracé du signal filtré
subplot(2,1,1)
title("Chebychev de type 2")
plot(1:length(y),y);
xlabel('Temps (échantillons)');
ylabel('Amplitude');

% Tracé du spectre d'amplitude du signal filtré
S = fft(y)/fe;
f = [0:length(y)-1]*(fe/length(S));
subplot(2,1,2)
plot(f, 20*log10(abs(S)));
axis([fmin fmax -200 0]);
xlabel('Fréquence (Hz)');
ylabel('Amplitude (dB)');



%%%%% Fenêtre de Kaiser %%%%%
ep = 1 - 10^(-Ap/20);
da = 10^(-(Aa+3.5)/20);

% Calcul de l'ordre et des coefficients du filtre de Kaiser
[n,wn,beta] = kaiserord([fbp fba fha fhp],[1 0 1],[ep da ep],fe);
h = fir1(n,wn,'stop',kaiser(n+1,beta));

% Affichage de l'ordre du filtre
disp(['Ordre du filtre de Kaiser : ', num2str(n)]);

f = linspace(fmin,fmax,samples);

% Calcul de la réponse fréquentielle en amplitude
H = freqz(h,1,2*pi*f/fe);
fig = figure;
set(fig, 'Name', 'Réponse freq filtre RIF', 'NumberTitle', 'off');
plot(f, 20*log10(abs(H)));
xlabel('Fréquence (Hz)');
ylabel('Amplitude (dB)');
legend('Kaiser');
grid on;

%%% Chebychev2 %%%
fig = figure;
set(fig, 'Name', 'Signal filtré & spectre amplitude RIF', 'NumberTitle', 'off');
% Tracé du signal filtré
subplot(2,1,1)
title("Chebychev de type 2")
plot(1:length(y),y);
xlabel('Temps (échantillons)');
ylabel('Amplitude');

% Tracé du spectre d'amplitude du signal filtré
S = fft(y)/fe;
f = [0:length(y)-1]*(fe/length(S));
subplot(2,1,2)
plot(f, 20*log10(abs(S)));
axis([fmin fmax -200 0]);
xlabel('Fréquence (Hz)');
ylabel('Amplitude (dB)');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Affichage des 2 réponses sur même graphe %%%
f = linspace(fmin,fmax,samples);
% Calcul des réponses fréquentielles en amplitude des 2 filtres
fig = figure;
set(fig, 'Name', 'RII vs RIF', 'NumberTitle', 'off');
plot(f, 20*log10(abs(Hc1)));
hold on;

plot(f, 20*log10(abs(H)));

xlabel('Fréquence (Hz)');
ylabel('Amplitude (dB)');
legend('RII','RIF');
grid on;
hold off;