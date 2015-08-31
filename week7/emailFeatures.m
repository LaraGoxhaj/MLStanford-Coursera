function x = emailFeatures(word_indices)
%EMAILFEATURES takes in a word_indices vector and produces a feature vector
%from the word indices
%   x = EMAILFEATURES(word_indices) takes in a word_indices vector and 
%   produces a feature vector from the word indices. 

% Total number of words in the dictionary
n = 1899;

x = zeros(n, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return a feature vector for the
%               given email. The variable word_indices contains the list
%		of indices of the words which occur in one email.
%              Your task is take to construct a binary feature vector
%              that indicates whether a particular word occurs.

x(word_indices) = 1;

% =========================================================================

end
