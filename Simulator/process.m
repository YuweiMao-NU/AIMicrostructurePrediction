%This code works for multiple runs of a particular process sequence: this can be
%single or combined applied force. 'param' file is used in every run
% see the instruction in the help file if loading conditions vary in each
% run

clear all; clc;

load newmesh
load Copper_Properties  % Loading cubic structure data

% Input ODFs for the first run

odf = (1/sum(volumefraction))*ones(145,1);

T1=table(odf);
writetable(T1,'Input_ODF.txt','WriteVariableNames',0); % saving as initial Input_ODF in the folder

param=zeros(1,8);
tmp = 0;
n=1;

for pa1=0:0.25:1
    for pa2=0:0.25:1
        for pa3=0:0.25:1
            for pa4=0:0.25:1
                for pa5=0:0.25:1
                    tmp = tmp+1;
                    odf = (1/sum(volumefraction))*ones(145,1);
                    T1=table(odf);
                    writetable(T1,'Input_ODF.txt','WriteVariableNames',0);

                    param(1, 1)=pa1;
                    param(1, 2)=pa2;
                    param(1, 3)=pa3;
                    param(1, 4)=pa4;
                    param(1, 5)=pa5;

                    T2=table(param);
                    writetable(T2,'param.txt','WriteVariableNames',0,'Delimiter','\t') % saving param file in the folder


                    odf_eachstep_total=zeros(145,10,n);      % Raw ODFs after all runs
                    odf_normalized_total_76=zeros(76,10,n);   % Normalized independent (76) ODFs after all runs
                    odf_normalized_total_145=zeros(145,10,n); % Normalized independent and dependent (145) ODFs after all runs


                    system("/home/ymt1957/wine-dirs/wine64-build/wine /data/ymt1957/processing/Simulator/app.exe"); % Command for process running

            %       system('app.exe'); % Command for process running

                    % Initialize matrix A to store the last column from each file
                    A = zeros(145, 10); % Since we are taking 145 lines from 10 files

                    % Loop over all 10 files
                    for fileIdx = 1:10
                        % Generate the file name, e.g., ODFField0001.out, ODFField0002.out, etc.
                        fileName = sprintf('ODFField%04d.out', fileIdx);

                        % Open the file for reading
                        fileID = fopen(fileName, 'r');

                        % Read the entire file into a cell array, one line per cell
                        fileContent = textscan(fileID, '%s', 'Delimiter', '\n');
                        fileContent = fileContent{1}; % Extract the content from the cell array

                        % Close the file
                        fclose(fileID);

                        % Extract lines from 5 to 149 (these lines are stored in cell array 5:149)
                        selectedLines = fileContent(5:149);

                        % Loop through each of these lines to extract the last column
                        for lineIdx = 1:length(selectedLines)
                            % Split the line into individual columns based on whitespace
                            columns = strsplit(selectedLines{lineIdx});

                            % Convert the last column to a number and store in matrix A
                            A(lineIdx, fileIdx) = str2double(columns{end});
                        end
                    end

                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    % Initialize B with 76 rows and 10 columns
                    B = zeros(76, 10);

                    % Load the mapping from the 'mapping.txt' file
                    % Assuming the mapping file has the format like "B(1) = A(57)", "B(2) = A(13)", etc.

                    fileID = fopen('mapping.txt', 'r');
                    mappingData = textscan(fileID, 'B(%d) = A(%d)', 'Delimiter', '\n');
                    fclose(fileID);

                    % Extract the row indices for B and A from the mapping data
                    B_indices = mappingData{1};
                    A_indices = mappingData{2};

                    % Map the corresponding rows from A to B using the mapping
                    for i = 1:length(B_indices)
                        B(B_indices(i), :) = A(A_indices(i), :);
                    end
                    odf_normalized_total_76 = B;
                    path = ['data_new/', num2str(pa1) '_',num2str(pa2),'_',num2str(pa3),'_', num2str(pa4),'_', num2str(pa5), '.mat']
                    save(path, 'odf_normalized_total_76');
                end
            end
        end
    end
end
                    



