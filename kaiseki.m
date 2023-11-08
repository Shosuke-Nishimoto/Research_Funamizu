list = dir('*.mat');
TrueSide = [];
Outcome = [];
EvidenceStrength = [];
nTrials = [];
for i = 2:length(list)
    load(list(i).name);
    TrueSide = [TrueSide saveBlock.TrueSide];
    Outcome = [Outcome saveBlock.Outcome];
    EvidenceStrength = [EvidenceStrength saveBlock.EvidenceStrength];
    nTrials = [nTrials saveBlock.nTrials];
end

global bin_size %this is for convenience
global BpodSystem

BpodSystem.TrueSide = TrueSide;
BpodSystem.Outcome = Outcome;
BpodSystem.EvidenceStrength = EvidenceStrength;
BpodSystem.nTrials = max(nTrials-50);

BpodSystem.ProtocolFigures.PsychoPlotFig = figure('Position', [1200 100 500 300],...
    'name','Psychometric plot','numbertitle','off', 'MenuBar', 'none', 'Resize', 'off');
BpodSystem.GUIHandles.PsychoPlot = axes('Position', [.2 .25 .75 .65]);
PsychoPlot_N_all(BpodSystem.GUIHandles.PsychoPlot,'init')  %set up axes nicely

UpdatePsychoPlot();
exportgraphics(BpodSystem.ProtocolFigures.PsychoPlotFig,'2023_01_23.jpg')

%%
function UpdatePsychoPlot()
global BpodSystem
TrueSide = BpodSystem.TrueSide;
Outcome = BpodSystem.Outcome;
EvidenceStrength = BpodSystem.EvidenceStrength;
nTrials = BpodSystem.nTrials;
PsychoPlot_N_all(BpodSystem.GUIHandles.PsychoPlot,'update',nTrials,TrueSide,Outcome-1,EvidenceStrength);
return
end