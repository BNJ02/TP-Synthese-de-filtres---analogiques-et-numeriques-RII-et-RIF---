function [gam,t] = gamme(duree,fe)

% [do re mi fa sol la si doo]
freqnotes = [262 294 330 349 392 440 494 523];
t = 0:1/fe:duree;
gam = [];
for note=1:8, 
    gam = [gam, sin(2*pi*freqnotes(note)*t)];
end
N = length(gam);
t = (0:N-1)/fe;



