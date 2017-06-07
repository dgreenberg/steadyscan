the mexfile in this repository (align_group_to_template.mexa64) is compiled under ubuntu 16.04 x86_64 with cuda 8.0 for nvidia cards with compute capability 3.5.
in order to compile the mexfile (align_group_to_template), the c++ sourcecode is supplied in setup/source
a validate script (setup/validate.m) can be used to check dependencies (MATLAB® mex and Eigen https://bitbucket.org/eigen/eigen )
the script also creates a new makefile if all dependencies are available.

a backup makefile is supplied in case the script is unable to complete (setup/source/mk/makefile_backup)
cuda and mex dependencies need to be changed manually

to align an image sequence to a template, the simplest call is
[~, ~, ~, ~, ~, ~, ~, data_corrected, ~] = steady_scan(data, template_image, [], 0, [], [], [], [], '' );

the settings struct (6. parameter) can be partially filled; missing settings are set to default values
all values and their defaults can be found at the bottom of the script (steady_scan.m) in the function default_alignment_settings.
