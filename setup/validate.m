function settings = validate(settings)
%
%

    %%%%%%%%%%%%%%%%%%%%%%%%%%%DEFAULTS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if ~exist('settings', 'var') || isempty(settings), settings = struct(); end    
    if ~isfield(settings, 'allow_32bit_compilation'), settings.allow_32bit_compilation = false; end
    
    if ~isfield(settings, 'char_ok'), settings.char_ok = '[✓]'; end
    if ~isfield(settings, 'char_nok'), settings.char_nok = '[x]'; end
    if ~isfield(settings, 'char_ncompile'), settings.char_ncompile = '[!]'; end
    if ~isfield(settings, 'char_unknown'), settings.char_unknown = '[?]'; end
    
    settings.char_matlab = 'MATLAB®';
    settings.url_cuda = 'https://developer.nvidia.com/cuda-downloads';
    settings.url_cc = 'https://developer.nvidia.com/cuda-gpus';
    settings.url_eigen = 'https://bitbucket.org/eigen/eigen/downloads/';
    settings.url_repo = 'https://github.com/dgreenberg/steadyscan';
    
    checks = {@check_os, @check_executable, @check_helper, @check_nvidia_toolkit, @check_devices, @check_source, @check_makefile, @check_manual_steps};
    
    settings.cuda_min_version = 6.0;
    settings.cuda_min_compute_capability = 3.5;
    
    settings.can_compile = true;
    settings.can_execute = true;    
    
    settings.query_compute_capability_available = false;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    fprintf('%s: configuration ok\n', settings.char_ok);
    fprintf('%s: missing; unable to execute\n', settings.char_nok);
    fprintf('%s: warning; unable to compile (but able to execute if mex-file present)\n', settings.char_ncompile);
    fprintf('%s: unable to query;assumed ok\n', settings.char_unknown);
    disp('-------------------------------------------------------------------------');
        
    for i=1:numel(checks)
        settings = checks{i}(settings);
    end
    
    settings = orderfields(settings);
    
    if settings.can_execute
        fprintf('able to execute (if mexfile present): %s\n', 'true');
    else        
        fprintf('able to execute (if mexfile present): %s\n', 'false');
    end
    
    if settings.can_compile
        fprintf('able to compile: %s\n', 'true');
    else        
        fprintf('able to compile: %s\n', 'false');
    end
end

function settings = check_os(settings)
    settings.ispc = ispc(); 
    if settings.ispc
        settings.can_compile = false; 
        fprintf('%s windows is not supported by the compilation script\n', settings.char_ncompile);
        fprintf('   ->suggested action: install x86_64 unix operating system\n');
    else
        fprintf('%s the operating system is supported by the compilation script\n', settings.char_ok);
    end
end

function settings = check_executable(settings)
    mx = mexext;
    settings.is64 = strcmp( '64', mx( (end-1):end ) ) ~= 0;
    
    if settings.is64
        fprintf('%s %s is running in 64bit mode\n', settings.char_ok, settings.char_matlab); 
    else
        settings.can_compile = settings.allow_32bit_compilation;
        if settings.can_compile
            fprintf('%s %s is not running in 64bit mode but 32bit compilation is allowed in settings', settings.char_ok, settings.char_matlab);
        else
            fprintf('%s %s is not running in 64bit mode (this might work but has not been tested in any way)\n', settings.char_ncompile, settings.char_matlab);
            fprintf('   ->suggested action: run 64bit version of %s\n', settings.char_matlab);
            fprintf('    to allow 32bit compilation pass a struct with ''allow_32_bit_compilation'' set to true\n');
        end
    end
end

function settings = check_helper(settings)
    if exist('query_compute_capability', 'file') == 3
        fprintf('%s query compute capability helper available\n', settings.char_ok);
        settings.query_compute_capability_available = true;
    else
        fprintf('%s query compute capability helper not available. assuming all reported devices fulfill the requirements (%.1f)\n', settings.char_ncompile, settings.cuda_min_compute_capability);
        fprintf('   ->suggested action: download ''query_compute_capability'' mexfile from repository into ''%s''\n', pwd);
        settings.query_compute_capability_available = false;
    end
end

function path = get_executable_path(name)
    if ~exist('name', 'var') || isempty(name)
        path = '';
        return;
    end
        
    %first check in ./
    if ispc()
        path = [pwd '\' name]; 
    else
        path = [pwd '/' name]; 
    end
        
    if exist(path, 'file')
        %path exists but it might be a symbolic link (shortcut)
        if ispc()
            %on windows we can check using the lnk extension
            if exist([path '.lnk'], 'file')
               %if its exist its a symbolic link 
               %TODO: resolve link; it may be needed for the compilation script to support windows; or is it?
            else
                %its not otherwise
                return;                
            end            
        else
            %on unix we can use readlink
           [status, outpt] = system(['readlink ' path]);
            if status == 0 
                %if readlink succeeds its a symbolic link
                path = outpt;
            else
                %its not otherwise
                return;
            end
        end
    else
        %no file found in ./
        %check in environment
        if ispc()
            %no support for windows
            path = name;
            return;
        else
           [status, outpt] = system(['which ' name]);
           if status == 0
              %executable available in environment
              path = outpt(1:end-1);
           else
              %executable unavailable
              return;
           end
        end        
    end
    
end

function settings = check_nvidia_toolkit(settings)
    %check nvidia-smi; used for gpu detection
    %try to detect path if none supplied
    if ~isfield(settings, 'exec_nvidia_smi'), settings.exec_nvidia_smi = get_executable_path('nvidia-smi'); end
    
    %this will fail with 'command not found:  -L' if exec_nvidia_smi is empty
    [status, outpt] = system( [settings.exec_nvidia_smi ' -L'] );
    if status ~= 0
        %no nvidia-smi no execution (or compilation)
        settings.can_compile = false;
        settings.can_execute = false;
        fprintf('%s unable to invoke nvidia-smi (%s)\n', settings.char_nok, settings.exec_nvidia_smi);
        fprintf('   ->suggested actions:\n');
        fprintf('   a) supply a valid path to nvidia-smi as member ''exec_nvidia_smi'' in settings argument\n');
        fprintf('   b)\n');
        fprintf('      1.) download and install current cuda toolkit from %s\n', settings.url_cuda);
        fprintf('      2.) make sure nvidia-smi is invokable from the current path (%s)\n', pwd);
    else
        fprintf('%s able to invoke nvidia-smi\n', settings.char_ok);
    end

    %save the output from nvidia-smi for check_devices
    gpus = strsplit(outpt, 'GPU ');
    settings.detected_gpu_devices = char({});
    for k=2:size( gpus, 2 )
        name = strsplit( gpus{k}, ' (' );          
        settings.detected_gpu_devices = cat( 1, settings.detected_gpu_devices, name( 1 ) ); 
    end
    
    %check nvcc; used for compilation
    %try to detect path if none supplied
    if ~isfield(settings, 'exec_nvcc'), settings.exec_nvcc = get_executable_path('nvcc'); end
    
    %this will fail with 'command not found:  --version' if exec_nvcc is empty
    [status, outpt] = system( [settings.exec_nvcc ' --version'] );
    if status ~= 0
        settings.can_compile = false;
        fprintf('%s unable to invoke nvcc (%s)\n', settings.char_ncompile, settings.exec_nvcc);
        fprintf('   ->suggested actions:\n');
        fprintf('   a) supply a valid path to nvcc as member ''exec_nvcc'' in settings argument\n');
        fprintf('   b)\n');
        fprintf('      1.) download and install current cuda toolkit from %s\n', settings.url_cuda);
        fprintf('      2.) make sure nvcc is invokable from the current path (%s)\n', pwd);
    else
        fprintf('%s able to invoke nvcc\n', settings.char_ok);
    end
    
    %validate cuda version
    %FIXME: assuming cuda toolkit is always capable of compiling for all present gpu devices
    if status ~= 0
        fprintf('%s unknown cuda version (nvcc not detected, min. V%.1f required)\n', settings.char_unknown, settings.cuda_min_version);
    else        
        osplit = strsplit(outpt, ', ');
        osplit = strsplit(osplit{2}, ' ');

        cuda_version = str2double(osplit{2});
        if cuda_version >= settings.cuda_min_version
            fprintf('%s supported cuda version (V%s detected, min. V%.1f required)\n', settings.char_ok, osplit{2}, settings.cuda_min_version);
        else
            settings.can_execute = false;
            fprintf('%s unsupported cuda version (V%s detected, min. V%.1f required)\n', settings.char_nok, osplit{2}, settings.cuda_min_version);            
            fprintf('   ->suggested actions:\n');
            fprintf('   1.) download and install current cuda toolkit from %s\n', settings.url_cuda);
            fprintf('   2.) make sure nvcc is invokable from the current path (%s)\n', pwd);
        end
    end
    
    
    %validate g++ availability and version
    %FIXME: there may be support for g++>4.7 in future cuda releases
    
    if ~isfield(settings, 'exec_gpp')    
        %if nvcc is called in environment, exec_nvcc is 4 characters long
        %TODO: make sure caller specifies ./nvcc
        if size(settings.exec_nvcc, 1) == 4
           %use which to find path of nvcc 
            [status, outpt] = system('which nvcc');
            assert(status == 0, 'unable to locate nvcc when we should;executed ''which''; status code %i; return value %s', status, outpt);
            settings.exec_gpp = [outpt(1:(end-4)) 'g++'];
        else
            %use settings.exec_nvcc to build settings.exec_gpp
            settings.exec_gpp = [settings.exec_nvcc(1:(end-4)) 'g++'];
            %make sure we don't have a symlink
            
           [status, outpt] = system(['readlink ' settings.exec_gpp]);
            if status == 0 
                %if readlink succeeds its a symbolic link
                settings.exec_gpp = outpt(1:end-1);
            end
        end        
    end
    
    %settings.exec_gpp now contains the path to the g++ executable
    %check if its version 4.7
    [status, outpt] = system([settings.exec_gpp ' --version']);
    if status ~= 0
        settings.can_compile = false;
        fprintf('%s unable to invoke g++ (%s)\n', settings.char_ncompile, settings.exec_gpp);
        fprintf('   ->suggested actions:\n');
        fprintf('   a) supply a valid path to g++ as member ''exec_gpp'' in settings argument\n');
        fprintf('   b)\n');
        fprintf('      1.) download and install g++\n');
        fprintf('      2.) make sure g++ is invokable from nvccs path (%s)\n', settings.exec_nvcc(1:end-5) );
    else
        %expecting first line to be something like 
        %g++ (Ubuntu 5.4.0-6ubuntu1~16.04.4) 5.4.0 20160609
        osplit = strsplit(outpt, ') ');
        osplit = strsplit(osplit{2}, '.'); %5, 4, 0 20160609
        gppver = str2double(osplit{1})*100 + str2double(osplit{2});
        if gppver == 407
            fprintf('%s g++ version compatible with nvcc (V%s detected, V%.1f required)\n', settings.char_ok, char([osplit{1} '.' osplit{2}]), 4.7);
        else 
            settings.can_compile = false;
            fprintf('%s g++ version incompatible with nvcc (V%s detected, V%.1f required)\n', settings.char_ncompile, char([osplit{1} '.' osplit{2}]), 4.7);
            fprintf('   ->suggested actions:\n');
            fprintf('   a) supply a valid path to g++ V4.7 as member ''exec_gpp'' in settings argument\n');
            fprintf('   b)\n');
            fprintf('      1.) download and install g++ V4.7\n');
            fprintf('      2.) make sure g++ V4.7 is invokable from nvccs path (%s)\n', settings.exec_nvcc(1:end-5) );
        end
        
    end
        
    
    %validate existing mex binary
    
    if ~isfield(settings, 'exec_mex'), settings.exec_mex = [matlabroot '/bin/mex']; end
    
    status = exist(settings.exec_mex, 'file');
    
    if status == 0
        settings.can_compile = false;
        fprintf('%s unable to invoke mex (%s)\n', settings.char_ncompile, settings.exec_mex);
        fprintf('   ->suggested actions:\n');
        fprintf('      a) supply a valid path to mex as member ''exec_mex'' in settings argument\n');
        fprintf('      b) make sure mex is invokable from current directory (%s)\n', pwd);
    else
        fprintf('%s able to invoke mex\n', settings.char_ok);
    end
       
end

function settings = check_devices(settings)
%

    %assuming all cards fulfill required compute capability if the script is unable to check
    if ~settings.query_compute_capability_available
        settings.can_compile = false;
        settings.available_gpu_devices = settings.detected_gpu_devices;
        
        %print a warning about skipped requirement check
        fprintf('%sunable to verify compute capability requirements. assuming all detected devices fulfill the requirement\n', settings.char_unknown);
        return;
    end
    
    %query compute capabilities of all cards and validate against requirement
    settings.available_gpu_devices = char({});
    settings.available_gpu_capabilities = char({});

    for i=1:size(settings.detected_gpu_devices, 1)
       %get id from name
       gpu_id = strsplit(settings.detected_gpu_devices{i}, ':');
       
       %query compute capability from device
       ccstr = query_compute_capability(str2double(gpu_id{1}));
       cc = str2double(ccstr);
       %validate against requirement
       if cc >= settings.cuda_min_compute_capability
           %add to available devices if ok
          settings.available_gpu_devices = cat(1, settings.available_gpu_devices, settings.detected_gpu_devices{i}); 
          settings.available_gpu_capabilities = cat(1, settings.available_gpu_capabilities, ccstr);
       end
    end
    
       
    if size(settings.available_gpu_devices, 1) > 0
        fprintf('%s at least one supported gpu detected (%i devices with compute capability >= %.1f)\n', settings.char_ok, size(settings.available_gpu_devices, 1), settings.cuda_min_compute_capability);
    else
        fprintf('%s no supported gpu detected (0 devices with compute capability >= %.1f)\n', settings.char_nok, settings.cuda_min_compute_capability);
        fprintf('   ->suggested actions:\n');
        fprintf('   a) validate that the device driver is running and up to date\n');
        fprintf('   b) install a card with compute capability >= %.1f\n', settings.cuda_min_compute_capability);
        fprintf('      a list with available cards and their compute capabilities can be found at %s\n', settings.url_cc);
    end
end

function settings = check_source(settings)
%
%
    sourceok = true;
        
    %check for source directory    
    if ~isfield(settings, 'src_main')
        if ~exist('./source', 'dir')
            settings.can_compile = false;
            fprintf('%s unable to locate source directory (%s/source)\n', settings.char_ncompile, pwd); 
            fprintf('   ->suggested action: download the source directory from the repository (%s) into %s\n', settings.url_repo, pwd);
            sourceok = false;
        else
            fprintf('%s source directory found\n', settings.char_ok);
            if ispc()
                settings.src_main = [pwd '\source'];
            else
                settings.src_main = [pwd '/source'];
            end
        end
    else
       if exist( settings.src_main, 'dir')
           fprintf('%s source directory detected (%s)\n', settings.char_ok, settings.src_main);
       else
            sourceok = false;
            settings.can_compile = false;
            fprintf('%s invalid source directory specified (%s is not accessable)\n', settings.char_ncompile, settings.src_main);
       end
    end
    
    %check for eigen repository
    eigenok = true;
    if ~isfield(settings, 'src_eigen')
        files_in_setup = dir;
        for i=1:numel(files_in_setup)
            if size(files_in_setup(i).name, 2) > 11 && all(files_in_setup(i).name(1:11) == 'eigen-eigen')
                if ispc()   
                    settings.src_eigen = [pwd '\' files_in_setup(i).name];                 
                else
                    settings.src_eigen = [pwd '/' files_in_setup(i).name];
                end
                
                break
            end
        end
        
        if ~isfield(settings, 'src_eigen')
            eigenok = false;
            fprintf('%s unable to locate eigen repository\n', settings.char_ncompile); 
            fprintf('   ->suggested actions:\n');
            fprintf('      a) supply a valid path to the eigen repository as member ''src_eigen'' in settings argument\n');
            fprintf('      b) download eigen repository (from %s) into %s\n', settings.url_eigen, pwd);  
        else
            fprintf('%s eigen repository detected (%s)\n', settings.char_ok, settings.src_eigen);
        end
    else
        if exist(settings.src_eigen, 'dir')
           fprintf('%s eigen repository detected (%s)\n', settings.char_ok, settings.src_eigen);
        else
            eigenok = false;
            settings.can_compile = false;
            fprintf('%s invalid eigen repository specified (%s is not accessable)\n', settings.char_ncompile, settings.src_eigen);
        end
    end
    
    
    %check for mex header
    if ~isfield(settings, 'src_mex')
        if ispc()
            settings.src_mex = [matlabroot '\extern\include\mex.h'];
        else
            settings.src_mex = [matlabroot '/extern/include/mex.h'];
        end        
    end
    
    if exist(settings.src_mex, 'file')
        mexok = true;
        fprintf('%s able to access mex header file\n', settings.char_ok);
    else
        settings.can_compile = false;
        mexok = false;
        fprintf('%s unable to access mex header file (%s)\n', settings.char_ncompile, settings.src_mex);
    end
    
    %update localdependencies header
    if sourceok && eigenok && mexok
        f = fopen([settings.src_main '/localdependencies.h'], 'w');
        fprintf(f, '#ifndef LOCALDEPENDENCIES_H_\n#define LOCALDEPENDENCIES_H_\n\n');
        fprintf(f, ' #ifdef _MATLAB\n');
        fprintf(f, '  #define MEXINCLUDE "');
        fprintf(f, settings.src_mex);
        fprintf(f, '"\n #endif\n');
        fprintf(f, ' #define EIGENINCLUDE "');
        fprintf(f, settings.src_eigen);
        fprintf(f, '/Eigen/Sparse"\n\n#endif');
        fclose(f);
        
        fprintf('%s source ok; updated local dependencies header\n', settings.char_ok);
    else
        fprintf('%s unable to update local dependencies header\n', settings.char_ncompile);
        fprintf('   ->required steps for compilation:\n');
        
        
        if sourceok
            sourcedir = settings.src_main;
        else
            sourcedir = '*PATH_TO_SOURCE_DIRECTORY*';
        end
        
        if eigenok
            eigendir = settings.src_eigen;
        else
            eigendir = '*PATH_TO_YOUR_EIGEN_DIRECTORY*';
        end
        
        if mexok 
            mexheader = settings.src_mex;
        else
            mexheader = '*PATH_TO_YOUR_MEX_HEADER.h*';
        end
        
        if ispc()            
            fprintf('      in file "%s\\localdependencies.h" update:\n', sourcedir);
            fprintf('Line %i (%s) to %s\n', 5, '#define MEXINCLUDE "/usr/local/MATLAB/R2014b/extern/include/mex.h"', mexheader);
            fprintf('Line %i (%s) to %s\n', 7, '#define EIGENINCLUDE "/data/steadyscan/setup/eigen-eigen-043c847d2c34/Eigen/Sparse"', ['"' eigendir '\\Eigen\\Sparse"']);
        else
            fprintf('      in file "%s/localdependencies.h" update:\n', sourcedir);
            fprintf('Line %i (%s) to %s\n', 5, '#define MEXINCLUDE "/usr/local/MATLAB/R2014b/extern/include/mex.h"', mexheader);
            fprintf('Line %i (%s) to %s\n', 7, '#define EIGENINCLUDE "/data/steadyscan/setup/eigen-eigen-043c847d2c34/Eigen/Sparse"', ['"' eigendir '\\Eigen\\Sparse"']);
        end
    end    
end

function settings = check_makefile(settings)
%

    %this only makes sense when we can compile (know all the paths)
    if ~settings.can_compile || settings.ispc
       fprintf('%s unable to generate makefile\n', settings.char_ncompile); 
       fprintf('\t->suggested action:\n');
       fprintf('\ta) execute suggested actions\n');
       fprintf('\tb) use makefile_backup in %s\n', [settings.src_main '/mk']); 
       return;
    end
       
    
    f = fopen([settings.src_main '/mk/makefile'], 'w');
    %path to nvcc
    fwrite(f, 'NVCC = ');
    fwrite(f, settings.exec_nvcc);
    %flags and path to g++
    fwrite(f, [char(10) 'NVCCFLAGS = -D_FORCE_INLINES -D_MWAITXINTRIN_H_INCLUDED -O3 -Xcompiler -fPIC -std=c++11 --compile -ccbin=']);
    fwrite(f, settings.exec_gpp);
    
    %gencodes for all architectures
    for i=1:size(settings.available_gpu_capabilities, 1)
        ccshort = [settings.available_gpu_capabilities(i, 1) settings.available_gpu_capabilities(i, 3)];
        gencode = [' -gencode arch=compute_' ccshort ',code=sm_' ccshort ' -gencode arch=compute_' ccshort ',code=compute_' ccshort];
        fwrite(f, gencode);
    end
    
    %matlabroot
    fwrite(f, [char(10) 'MATLABROOT=']);
    fwrite(f, matlabroot);
    
    %static content
    fwrite(f, char(10));
    fwrite(f, fileread([settings.src_main '/mk/makefile_static']));
        
    fclose(f);
    
    fprintf('%s makefile generated (%s)\n!!\t please validate CUDA_INCLUDE_DIR and CUDA_LIB_DIR\n', settings.char_ok, [settings.src_main '/mk/makefile']);
end

function settings = check_manual_steps(settings)
    disp('-------------------------------------------------------------------------');
    fprintf('\n!!! the following steps are not automatically validated !!!\n');
    fprintf('1) %s mex default option uses the cxxflag std=c++11\n', settings.char_matlab);
    fprintf('2) in the cuda-toolkit edit cuda_fp16.h at lines 3068 and 3079:\n');
    fprintf('     manually changed "if (isinf(a)) {" to "if (std::isinf(a)) {"\n');
    fprintf('  OR manually changed "if (isinf(a)) {" to "if (::isinf(a)) {"\n');
    disp('-------------------------------------------------------------------------');
end
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
