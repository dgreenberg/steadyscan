function [dx_knots, dy_knots, iter_used, corr, failed, residrms, settings, data_corrected, mask] = ...
    steady_scan( data, template_image, data_extra_channels, n_extra_channels, frame_list, settings, est_dx_knots, est_dy_knots, wbstring ) 
%steady_scan
%
%Uses the GPU to correct horizontal motion artifacts in imaging acquired through a scanning process.
%Motion much faster than the imaging frame rate can be detected and corrected.
%
%Simplest possible use of this function:
%[~, ~, ~, ~, ~, ~, ~, data_corrected, ~] = steady_scan(data, template_image, [], 0, [], [], [], [], '' );
%
%
%Inputs:
%data
%   a) 3D array of double/single precision ( h x w x n ); n = number of frames
%   b) data function defined as value=datafcn(i); value is a frame ( h x w ) and i its 1-based index into the frame-sequence
%
%   h is the height of a frame; the number of scanlines
%   w is the width of a frame; the number of pixels per line
%   it is assumed that raster scanning has proceeded along each row of data
%
%template_image
%   an image to which the data will be aligned. 
%   it doesn't need to be the same size as each image in data.
%
%data_extra_channels
%   a) extra image channels to be corrected according to the motion detected
%      for the channel in data. must be a 4D array, with the first 3 dimensions matching data and
%      the 4th dimension for image channel
%   b) data function defined as value=datafcn(i, ch); value is a frame ( h x w ), i its 1-based index into the frame-sequence and ch the channel
%
%   can be empty if n_extra_channels is 0
%
%n_extra_channels
%   the number of additional channels to correct
%   if data_extra_channels is passed as 4D-array n_extra_channels is assumed to be equal to size(data_extra_channels, 4)
%
%   
%frame_list
%   a list of frames to be corrected in the imaging file. frame_list must not be empty when data is a function handle
%   if frame_list is empty is all frames are used
%
%settings
%   a structure whos fields contain parameters for motion correction.
%   possible values and their defaults can be found at the bottom of this script 
%   in the function default_alignment_settings
%
%est_dx_knots
%   2 dimensional array (nknots x nframes) containing an estimation for horizontal displacements
%
%est_dy_knots
%   2 dimensional array (nknots x nframes) containing an estimation for vertical displacements
%
%wbstring
%   text that will be shown in all waitbars

    %make sure mc-mex file is available
    assert(exist('align_group_to_template', 'file') == 3, 'motion correction mex file not found');
    
    %this function is used when data is actual data instead of a function handle
    function value = datafcn(i)
        value = data(:,:,i);
    end

    if isa(data,'function_handle')
        imagedatafcn = data;
        assert(~isempty(frame_list), 'frame_list must not be empty if data is passed by handle');
        nframes = numel(frame_list);
    else
        imagedatafcn = @datafcn;
        nframes = size(data, 3);
    end

    %extra channels
    function channeldata = channelfcn(i, ch)
        channeldata = data_extra_channels(:,:,i,ch);
    end

    if ~isempty(data_extra_channels) && isa(data_extra_channels, 'function_handle')
        extrachanneldatafcn = data_extra_channels;        
    else
        extrachanneldatafcn = @channelfcn;
    end
    
    %make sure we can append wbstring
    if ~exist('wbstring', 'var') || isempty(wbstring), wbstring = ''; end
        
    %settings
    [data, template_image, nlines, linewidth] = check_idata_inputsvars(data, template_image);
    if ~exist('settings','var'), settings = []; end
               
    if ~exist('frame_list','var') || isempty(frame_list),   frame_list = 1:nframes; end

    settings = init_settings(settings, template_image, nlines, linewidth);
    template_image = prefilter_foralignment(template_image, settings.pregauss, settings.mednorm);   
        
    
    %create set indices
    set_start = 1:settings.frames_per_set:nframes;
    set_start(end+1)=nframes+1;
    set_size = set_start(2:end)-set_start(1:end-1);
    
    %check if last set has less then gpu_min_frames_per_set frames
    %combine with previous set into two smaller ones of equal size
    if numel(set_size) > 1 && set_size(end) < settings.min_frames_per_set
        mid = ( set_size(end-1) + set_size(end) ) / 2;
        set_size( end - 1 ) = ceil(mid);
        set_size( end ) = floor(mid);        
    end
    
    
    [corr, iter_used, residrms, failed, dx_knots, dy_knots] = estimate_displacements(settings, imagedatafcn, template_image, est_dx_knots, est_dy_knots, wbstring, nframes, frame_list, set_start, set_size);
    
    
    
    %reconstruction
    
    
    %preallocate
    data_corrected = zeros([size(template_image) nframes 1+n_extra_channels]);
    mask = false([size(template_image) nframes 1+n_extra_channels]);
    
    %waitbar
    wb = waitbar(0,[wbstring char(10) 'reconstructing frames']);
    cleanup = onCleanup( @()( close( wb ) ) );
        
    for iset = 1:numel(set_size)
        %update progressbar
        waitbar((iset-1) / numel(set_size), wb);
        drawnow('expose');
        
        ii = set_start(iset):min( ( set_start(iset + 1)-1 ), nframes );
        setdata = prefilter_foralignment( cropimage(imagedatafcn( frame_list(ii) ), settings.cropping), settings.pregauss, settings.mednorm);
        [data_corrected(:,:,frame_list(ii),1), mask(:,:,frame_list(ii))] = invert_bilinear_sampling(setdata, settings, size(template_image), frame_list(ii), dx_knots, dy_knots );
    end
    
    if n_extra_channels ~= 0
        %close waitbar
        clear cleanup;
        clear wb;
    
        wb = waitbar(0,[wbstring char(10) 'reconstructing extra channels']);
        cleanup = onCleanup( @()( close( wb ) ) );
        for ch=1:n_extra_channels
            for iset = 1:numel(set_size)
                %update progressbar
                waitbar((iset-1) / (numel(set_size) * n_extra_channels), wb);
                drawnow('expose');

                ii = set_start(iset):min( ( set_start(iset + 1)-1 ), nframes );
                setdata = prefilter_foralignment( cropimage(extrachanneldatafcn( frame_list(ii), ch ), settings.cropping), settings.pregauss, settings.mednorm);
                [data_corrected(:, :, frame_list(ii), ch), mask(:, :, frame_list(ii), ch)] = invert_bilinear_sampling(setdata, settings, size(template_image), frame_list(ii), dx_knots, dy_knots );
            end
        end
    end
end

function [corr, iter_used, residrms, failed, dx_knots, dy_knots] = estimate_displacements(...
        settings, imagedatafcn, template_image, est_dx_knots, est_dy_knots, wbstring, nframes, frame_list, set_start, set_size)

    %preallocate outputs    
    [dx_knots, dy_knots] = deal(nan(settings.nknots, nframes));
    [corr, iter_used, residrms] = deal(nan(nframes, 1));
    failed = false(nframes, 1);
          
    %reshape indexing matrices into block order
    settings.prevknotind = reshape(settings.prevknotind, [], 1);
    settings.basex = reshape(settings.basex, [], 1);
    settings.basey = reshape(settings.basey, [], 1);  
    
    [tfrac, base_mask] = image2blockmatrix(settings.tfrac, settings.prevknotind, settings.nknots - 1);       
    basex = image2blockmatrix( settings.basex - 1, settings.prevknotind, settings.nknots - 1 );
    basey = image2blockmatrix( settings.basey - 1, settings.prevknotind, settings.nknots - 1 );
        
    points_per_block = histc( settings.prevknotind, 1:settings.nknots );
    min_points_per_block = points_per_block * settings.min_blockpoints_ratio;
        
    %gradients. NaNs are appended so they can be indexed the same as the template
    xgrad = [diff( template_image, 1, 2 )  nan( size( template_image, 1 ), 1 )];  
    ygrad = [diff( template_image, 1, 1 ); nan( 1, size( template_image, 2 ) )];
    
    %parameter estimation
    if ~isempty(est_dx_knots) && ~isempty(est_dy_knots)
        %TODO: maybe don't trust values of est_dx/y_knots ? => check correlation before alignment?
        assert( all( size( est_dx_knots ) == size( dx_knots ) ), 'est_dx is incorrectly sized' );
        assert( all( size( est_dy_knots ) == size( dy_knots ) ), 'est_dy is incorrectly sized' );
    else
        %if no est_given we have to use init_from zero
        est_dx_knots = zeros(size(dx_knots));
        est_dy_knots = zeros(size(dy_knots));
    end
    
    %gpu parameters
    %pass parameter over variables so that c++ code can reference
    %parameter names for error/warning messages
    group_size = size(basex, 2) * settings.gpu_frame_groupsize;
    max_iter = settings.max_iter;
    move_thresh = settings.move_thresh;
    halt_correlation = settings.haltcorr;
    correlation_increase_threshold = settings.gpu_correlation_increase_thresh;
    filter_strength = settings.gpu_filter_strength;
    max_filter_iterations = settings.gpu_filter_max_iterations;
    filter_deactivation_threshold = settings.gpu_filter_correlation_thresh;
    linesearch_reduction_multiplier = settings.linesearch_reduction_multiplier;     
    gpu_device_id = settings.gpu_device_id;
    filter_id = settings.gpu_filter_id;
    solver_id = settings.gpu_solver_id;
    %TODO: implement lambda
    lambda = 0;  
    
    wb = waitbar(0,[wbstring char(10) 'estimating displacements']);
    cleanup = onCleanup( @()( close( wb ) ) );
    
    %align each set seperately
    %last frame of set is estimated incorrectly
    %sets are extended by 1 frame, its estimations are discarded
    for iset = 1:numel(set_size)
        %update progressbar
        waitbar((iset-1) / numel(set_size), wb);
        drawnow('expose');
        
        %get frame indices
        ii = set_start(iset):min( ( set_start(iset + 1) ), nframes );
        
        %read set into memory
        setdata = prefilter_foralignment( imagedatafcn( frame_list(ii) ), settings.pregauss, settings.mednorm);
        nsetframes = size(setdata, 3);
        
        %concatenate all frames in current set to one
        bmData = zeros(size(basex, 1), size(basex, 2) * nsetframes);        
        for k=1:nsetframes
           fis = (size(basex, 2)) * (k-1) + 1; 
           fie = fis + size(basex, 2) - 1;
           bmData(:, fis:fie) = image2blockmatrix(cropimage(setdata(:,:,k), settings.cropping), settings.prevknotind, settings.nknots - 1);
        end
        
        flags = [settings.gpu_forced_precision settings.gpu_flags];
        
        %align set on gpu
        [set_dx_knots, set_dy_knots, frame_corr, set_iter_used, set_mask, set_residrms] = ...
            align_group_to_template( flags, template_image, xgrad, ygrad, tfrac, basex, basey, base_mask, bmData, est_dx_knots, est_dy_knots, ...
            group_size, nsetframes, min_points_per_block, max_iter, move_thresh, halt_correlation, correlation_increase_threshold, ...
            filter_strength, max_filter_iterations, filter_deactivation_threshold, lambda, linesearch_reduction_multiplier, solver_id, filter_id, gpu_device_id );
        %clear mexfile from memory
        clear align_group_to_template; 
            
        for setj = 1:( nsetframes - 1 )
            %index into frame_list for slice blockj of the current block
            j = ii(setj); 
            
            %copy dx/y_knots
            dx_knots(:, j) = set_dx_knots((setj-1) * settings.nknots + ( 1:settings.nknots ) );
            dy_knots(:, j) = set_dy_knots((setj-1) * settings.nknots + ( 1:settings.nknots ) );
            
            %copy correlation
            corr(j) = frame_corr(setj);
            
            %copy used iterations
            iter_used(j) = set_iter_used;
            
            %copy residrms            
            residrms(j) = set_residrms(setj);
            
            %reconstruct mask         
            frame_mask = set_mask(:,(setj-1) * (settings.nknots-1) + ( 1:(settings.nknots-1) ));
            blocksused = histc( settings.prevknotind( blockmatrix2image( frame_mask, settings.prevknotind, base_mask ) ), 1:settings.nknots - 1) > 0;
                                    
            %alignment failed if all blocks are estimated to be
            %outside of the template
            if ~any(blocksused)
                failed(j) = true;
                corr(j) = nan;
                continue;
            end
            
            %if part of the template was not used (e.g. partial vertical overlap),
            %then extend the first/last available displacement backward/forward in time             
            if ~blocksused(1) || ~blocksused(end)
                %FIXME should we recalculate correlation etc.?
                dx_knots(:, j) = extend_firstandlast(dx_knots(:, j), blocksused);
                dy_knots(:, j) = extend_firstandlast(dy_knots(:, j), blocksused);
            end
            
            failed(j) = ~( frame_corr( setj ) >= settings.corr_thresh ) || numel( blocksused ) / ( settings.nknots - 1 ) < settings.minblockfrac;
        end
        
        
    end
end

function [data_corrected, mask] = invert_bilinear_sampling(data, settings, corrected_data_size, frame_list, dx_knots, dy_knots)
    
    %[data_corrected, mask] = invert_bilinear_sampling(data, a, corrected_data_size, inpaint, aindlist)
    %note that aindlist is used to index arrays in a, but it is assumed that data has already been indexed
    %i.e. size(data, 3) == numel(aindlist)
    swwthresh = 0; %0.8;
    if ~exist('frame_list','var') || isempty(frame_list), frame_list = 1:size(data, 3); end    
    assert((isa(data,'double') || isa(data,'single')) && ~isa(data,'complex'), 'data must be real valued');
    
    npixels = prod(corrected_data_size);
    nframes = size(data, 3);
    assert(numel(frame_list) == size(data, 3), 'invalid input');
    basex = reshape(settings.basex, [], 1); basey = reshape(settings.basey, [], 1);
    tfrac = reshape(settings.tfrac, [], 1);
    prevknotind = reshape(settings.prevknotind, [], 1);
    tcomp = 1 - tfrac;
    %increment the linear system of constraints for each chunk (see comments in bilinear_map_linear_constraints)
    data_corrected =   nan([npixels nframes], class(data));
    mask           = false([npixels nframes]);
    
    % pixel aspect ratio is pixel width / pixel height
    % if pixels are n times wider than high, we penalize vertical differences n times less
    vfac = 1 / settings.ibs_template_pixel_aspect_ratio;
    regmat = imageadjdiffquadmat(corrected_data_size(2), corrected_data_size(1), vfac) * settings.ibs_regfac;
    
    %'constants' for readability
    A=1;%B=2;C=3;
    D=4;
    
    %not all pixels have a partner
    %e.g. left A does not have a B on the left side
    %-> index is set to one in these cases, the according interaction term is zero
    %=> size(IA,W) == size(IA)
    [I, Imat, M] = calc_indices(corrected_data_size);
    
    nframes = size(data, 3);
    n_iter_error = 1;
    
    for i = 1:nframes                  
        [wy, ww] = bilinear_map_linear_constraints( data(:,:,i), dx_knots(:, frame_list(i)), dy_knots(:, frame_list(i)), basex, basey, tfrac, tcomp, prevknotind, corrected_data_size(2), corrected_data_size(1));
         sww = full( sum(ww) );
         
        %pixels for which we have sufficient summed coefficients (all positive) in ww. we have no guarantee the whole system isn't underdetermined though.
        mask(sww > swwthresh, i) = true; 
        if ~any(mask(:, i)), continue; end                  
        
        iv = resample_frame(data(:,:,i), corrected_data_size, dx_knots(:, frame_list(i)), dy_knots(:, frame_list(i)), basex, basey, prevknotind, tfrac);
        mask(isnan(iv(:)), i) = false;
        
        %mask out pixels that are not estimated
        Q = ww + regmat;
        Lfull = full(wy);
        
        %mask out Q and L   
        Q(~mask(:,i), :) = 0;
        Q(:, ~mask(:,i)) = 0;
        Lfull(~mask(:,i)) = 0;
        
        %calculate all interaction terms
        [R, Rmat] = calc_interactions(Q, I, Imat, M);
        
        %split L in 4 parts
        Lf = cell(4, 1);
        L = cell(4, 1);
        for k=A:D
            Lf{k} = single(Lfull(I{k}));
            L{k} = Lf{k} ./ R{k}; 
            L{k}(isnan(L{k}) | isinf(L{k})) = 0;
        end
        
        Qe = Q(mask(:,i), mask(:,i));
        Le = single(Lfull(mask(:,i))).';
        
        %initialize x
        %(x = data_corrected(:, i))
        x = iv(:);
        xo = single(zeros(size(x)));
        
        x(isnan(x)) = 0;
        iter = int32(0);
                
        %initial error
        eo = single(double(x(mask(:,i)).') * Qe * double(x(mask(:,i))) - 2 * double(Le * x(mask(:,i))));
        
        while 1  
            %update x
            for k=A:D                                  
                x(I{k}) = L{k} - sum(x(Imat{k}) .* Rmat{k}, 2);     
            end            
            
            %check error
            iter = iter + 1;
            if mod(iter, n_iter_error) == 0
                e = single(double(x(mask(:,i)).') * Qe * double(x(mask(:,i))) - 2 * double(Le * x(mask(:,i))));

                if eo <= e, break; end

                %store for next iteration
                eo = e;   
                xo = x;            
            end
        end      
        
        %calculate the number of iterations which skip error checking
        n_iter_error = int32(max(1, min(10, sqrt(single(iter)))));
        
        xo(~mask(:,i)) = NaN;
        data_corrected(:,i) = xo;        
    end   
    
    data_corrected = reshape(data_corrected, [corrected_data_size nframes]);
    mask           = reshape(mask,           [corrected_data_size nframes]);
                
    if settings.ibs_inpaint, data_corrected = imgapfill_leastsquares(data_corrected); end
end

function GtG = imageadjdiffquadmat(w, h, vfac)
    %GtG = imageadjdiffquadmat(n, m, vfac)
    %returns a sparse matrix GtG such that for any w x h image y,
    %y(:)' * GtG * y(:)
    %is the sum of squares of adjacent pixel value differences in y.
    if ~exist('vfac', 'var'), vfac = 1; end
    npixels = w * h;
    [x, y] = meshgrid(1:w, 1:h);
    x = reshape(x, [], 1);
    y = reshape(y, [], 1);

    % indentify pairs of adjacent pixels, with each pair counted only once
    xpairs = [x x + 1; x     x];
    ypairs = [y     y; y y + 1];
    
    % make sure both pixel locations are inside the image boundaries
    pairmask = all(xpairs <= w & ypairs <= h, 2); 
    xpairs = xpairs(pairmask, :);
    ypairs = ypairs(pairmask, :);
    npairs = size(xpairs, 1);
    pixelindpairs = ypairs + h * (xpairs - 1);

    % mask indicating which pairs are vertically adjacent
    vpairs = xpairs(:, 1) == xpairs(:, 2); 
    % [1 -1] for horizontally adjacent pairs, [vfac -vfac] for vertically adjacent pairs
    coefs = (vfac .^ double(vpairs)) * [1 -1]; 

    %construct an npairs-by-npixels matrix G such that (G * a(:))' * (G * a(:)) will be the sum of square differences in adjacent pixels of a
    G = sparse([1:npairs, 1:npairs]', [pixelindpairs(:, 1); pixelindpairs(:, 2)], coefs, npairs, npixels);
    GtG   = G' * G;
end

function [wy, ww] = bilinear_map_linear_constraints(data, dx_knots, dy_knots, basex, basey, tfrac, tcomp, prevknotind, templatew, templateh)
    %[wy, ww] = bilinear_map_linear_constraints(data, dx_knots, dy_knots, basex, basey, tfrac, tcomp, prevknotind, templatew, templateh)

    %For each 4-pixel square of the template, combine the pixel values and fractional coordinates for the data pixels inside it, combining data from all frames,
    %to produce linear constraints on the unknown template values. These constraints, which are simply the rules of bilinear interpolation,
    %consist of sum of squares error between the observed data values and the predicted values produced by bilinear interpolation of the unknown
    %template at fractional coordinates.
    %The equations are linear in the unknown template, linear in the observed data, and bilinear in the fractional parts of the x- and y- coordinates.
    %the constraints will take the form ww * template == wy, where template is in vector form. ww will be constructed as a sparse matrix.

    % we can derive these constraints based on the following sort of argument:
    % e = (v1 * T - b1)^2 + (v2 * T - b2)^2
    % then setting de/dT to zero we get
    % (v1 * v1' + v2 * v2') * T = (b1 * v1 + b2 * v2)

    %compute subpixel locations in the template for each fluorescence observation in the data
    x = basex + dx_knots(prevknotind, :) .* tcomp + dx_knots(prevknotind + 1, :) .* tfrac;
    y = basey + dy_knots(prevknotind, :) .* tcomp + dy_knots(prevknotind + 1, :) .* tfrac;
    
    %mask out pixels off the edge of the template
    mask = (x > 1 + eps) & (x < templatew - eps) & (y > 1 + eps) & (y < templateh - eps); 
    
    x = x(mask);
    y = y(mask);
    x_int = floor(x);
    y_int = floor(y);
    x_frac = x - x_int;
    y_frac = y - y_int;
    clear x y; %save memory
    matind_corners = bsxfun(@plus, y_int + templateh * (x_int - 1), [0 1 templateh templateh + 1]);
    clear x_int y_int;
    x_frac_comp = 1 - x_frac;
    y_frac_comp = 1 - y_frac;
    w = [x_frac_comp .* y_frac_comp, ...
        x_frac_comp  .* y_frac, ...
        x_frac       .* y_frac_comp, ...
        x_frac       .* y_frac];
    clear x_frac y_frac x_frac_comp y_frac_comp;
    wyvals = bsxfun(@times, w, data(mask));
    clear mask;
    wwvals = [ ...
        bsxfun(@times, w,       w(:, 1  )) ...
        bsxfun(@times, w(:, 2), w(:, 2:4)) ...
        bsxfun(@times, w(:, 3), w(:, 3:4)) ...
        w(:, 4) .* w(:, 4) ];
    clear w;
    %sparse() adds up input elements with the same row/column indices
    wy = sparse(matind_corners(:), 1, wyvals(:), templateh * templatew, 1);
    clear wyvals;
    corner1ind = [1 1 1 1 2 2 2 3 3 4];
    corner2ind = [1 2 3 4 2 3 4 3 4 4];
    matind_corner1 = matind_corners(:, corner1ind);
    matind_corner2 = matind_corners(:, corner2ind);
    clear matind_corners;
    offdiag = corner1ind ~= corner2ind;
    wwvals_all          = [wwvals(:);         reshape(wwvals(:,offdiag),  [], 1)];
    matdind_corner1_all = [matind_corner1(:); reshape(matind_corner2(:, offdiag), [], 1)];
    matdind_corner2_all = [matind_corner2(:); reshape(matind_corner1(:, offdiag), [], 1)];
    clear matind_corner1 matind_corner2 wwvals;
    %sparse() adds up input elements with the same row/column indices
    ww = sparse(matdind_corner1_all, matdind_corner2_all, wwvals_all, templateh * templatew, templateh * templatew); 
end

function [R, Rmat] = calc_interactions(Q, I, Imat, M)
    %'constants' for readability
    A=1;%B=2;C=3;
    D=4;
    N=1;%NE=2;E=3;SE=4;S=5;SW=6;W=7;
    NW=8;

    R = cell(4, 1);
    Rmat = cell(4, 1);
    
    qrows = int64(size(Q, 1));
    
    for k=A:D
        Rmat{k} = single(zeros(size(I{k}, 1), NW - N + 1)); 
        R{k} = single(full(Q(I{k} + (I{k} - 1) * qrows)));
        
        mask = R{k} ~= 0;
        
        for j=N:NW
            Rmat{k}(mask, j) = single((full(Q(I{k}(mask) + (Imat{k}(mask, j) - 1) * qrows))) ./ R{k}(mask)) .* M{k}(mask, j);
        end
    end    
end

function b = imgapfill_leastsquares(a)
    [h, w, n] = deal(size(a, 1), size(a, 2), size(a, 3));
    npixels = w * h;
    a = reshape(a, [npixels n]);
    mask = ~isnan(a);
    if all(mask)
        b = reshape(a, [h w n]);
        return;
    end
    GtG = imageadjdiffquadmat(w, h);
    %construct a matrix J such that the missing values c of a can be filled in by setting a -> a0 + J * c
    b = double(a);
    for k = 1:n
        if ~any(mask(:, k)), continue; end
        nmissing = sum(~mask(:, k));
        a0 = b(:, k);
        a0(~mask(:, k)) = 0;
        J = sparse(find(~mask(:, k)), 1:nmissing, 1, npixels, nmissing);
        %determine matrices Q, L such that the sum of square adjacent differences will be c' * Q * c + 2 * L * c + constant
        %FIXME use an indexing operation instead
        JtGtG = J' * GtG; 
        %FIXME use an indexing operation instead
        Q    = JtGtG * J;   
        Lt   = JtGtG * sparse(a0);
        
        %solve for c
        c = full(-(Q \ Lt)); 
        %fill in the missing values
        b(~mask(:,k), k) = c; 
    end
    b = reshape(b, [h w n]);
    if isa(a, 'integer')
        b = round(b);
    end
    b = cast(b, class(a));
end

function frame_corrected = resample_frame(data, corrected_data_size, dx_knots, dy_knots, basex, basey, prevknotind, tfrac)
    tcomp = 1 - tfrac;

    xpixelposition = basex + dx_knots(prevknotind) .* tcomp + dx_knots(prevknotind + 1) .* tfrac;
    ypixelposition = basey + dy_knots(prevknotind) .* tcomp + dy_knots(prevknotind + 1) .* tfrac;

    d = single(zeros(corrected_data_size));
    n = single(zeros(corrected_data_size));
    [d, n]= pushgrid(d(:,:), xpixelposition, ypixelposition, data, n(:,:));
    frame_corrected = d ./ n;
end

%add d to z in a grid-weighted way at (x,y) and record it in n
function [z,n] = pushgrid( z, x, y, d, n ) 
    nrows_template = size(z,1);
    ncols_template = size(z,2);

    mask = (x >= 1) & (x < ncols_template) & (y >= 1) & (y < nrows_template);
    x = x(mask);
    y = y(mask);
    d = d(mask);

    x_int = floor(x); y_int = floor(y);
    x_frac = x - x_int; y_frac = y - y_int;
    matind = y_int + nrows_template * (x_int - 1);

    %"bilinear-like" interpolation:
    z(matind) = z(matind) + (1 - x_frac) .* (1 - y_frac) .* d;
    z(matind + 1) = z(matind + 1) + (1 - x_frac) .* y_frac .* d;
    z(matind + nrows_template) = z(matind + nrows_template) + x_frac .* (1 - y_frac) .* d;
    z(matind + nrows_template + 1) = z(matind + nrows_template + 1) + x_frac .* y_frac .* d;

    %keeping track of denominator weights for final normalization:
    n(matind + 1) = n(matind + 1) + (1 - x_frac) .* y_frac;
    n(matind + nrows_template) = n(matind + nrows_template) + x_frac .* (1 - y_frac);
    n(matind) = n(matind) + (1 - x_frac) .* (1 - y_frac);
    n(matind + nrows_template + 1) = n(matind + nrows_template + 1) + x_frac .* y_frac;
end

function [y, bmmask] = image2blockmatrix(x, blockind, nblocks)
    %[y, bmmask] = image2blockmatrix(x, blockind, nblocks)
    %for an image grouped into blocks of pixels, creates a matrix with one column per block
    %and a mask indicating which extra values for each column should be ignored
    if ~exist('nblocks','var'), nblocks = max(blockind(:)); end
    pixels_per_block = histc(blockind(:), 1:nblocks);
    y = cast(zeros(max(pixels_per_block), nblocks), class(x)); %x can be of class logical
    bmmask = false(size(y));
    for b = 1:nblocks
        y(1:pixels_per_block(b), b) = x(blockind == b);
        bmmask(1:pixels_per_block(b), b) = true;
    end
end

function y = extend_firstandlast(x, mask)
    if ~any(mask(:)), return; end
    
    first = find(mask, 1);
    last  = find(mask, 1, 'last');
    y = x;
    y(1:first - 1) = x(first);
    y(last + 1:end) = x(last);
end

function data = prefilter_foralignment(data, gsd, mednorm)
    sD = [size(data, 1) size(data, 2) size(data, 3)];
    if gsd > 0
        L = ceil(sqrt((2 * gsd ^ 2) * log(10000 / (sqrt(2 * pi) * gsd))));
        sbig = sD(1:2) + L;
        [x, y] = meshgrid(1:sbig(2),1:sbig(1));
        g2d = exp( - (min(x - 1, sbig(2) + 1 - x) .^ 2 + min(y - 1, sbig(1) + 1 - y).^2) / (2 * gsd ^ 2) );
        g2d = g2d / sum(sum(g2d));
        g2d = fft2(g2d,sbig(1),sbig(2));
        for k = 1:size(data,3)
            u = real(ifft2(fft2(data(:,:,k),sbig(1),sbig(2)) .* g2d));
            data(:,:,k) = u(1:sD(1),1:sD(2));
        end
    end
    if mednorm
        data_median = median(reshape(data, [], sD(3)), 1); 
        data = bsxfun(@rdivide, data, reshape(data_median, [1 1 sD(3)]));
    end
end

function [I, Imat, M] = calc_indices(corrected_data_size)    
    A=1;B=2;C=3;D=4;
    N=1;%NE=2;E=3;SE=4;S=5;SW=6;W=7;
    NW=8;

    f = {@find_N @find_NE @find_E @find_SE @find_S @find_SW @find_W @find_NW};
    imat = repmat([A B;C D], ceil(corrected_data_size(1) / 2), ceil(corrected_data_size(2) / 2));
    imat = imat(1:corrected_data_size(1), 1:corrected_data_size(2));
    
    corrected_data_size = int64(corrected_data_size);
    npixels = corrected_data_size(1) * corrected_data_size(2);
    
    ind = (1:npixels).';
    
    I = cell(4, 1);
    Imat = cell(4, 1);
    
    M = cell(4, 1);
    
    for k=A:D
       I{k} = int64(ind(imat == k)); 
    end
    
    for k=A:D
       Imat{k} = int64(zeros(size(I{k}, 1), NW-N+1));
       M{k} = false(size(Imat{k}));
       for j=N:NW
          [Imat{k}(:,j), M{k}(:,j)] = f{j}(I{k}, corrected_data_size(1), npixels); 
       end
    end
end

function [I, M] = find_N(i, nrows, ~)
    I = i - 1;
    M = mod(I, nrows) ~= 0;
    I(~M) = 1;
end

function [I, M] = find_NE(i, nrows, npixels)
    I = i + nrows - 1;
    M = mod(I, nrows) ~= 0 & I < npixels;
    I(~M) = 1;
end

function [I, M] = find_E(i, nrows, npixels)
    I = i + nrows;
    M = I <= npixels;
    I(~M) = 1;
end

function [I, M] = find_SE(i, nrows, npixels)
    I = i + nrows + 1;
    M = mod(i, nrows) ~= 0 & I < npixels;
    I(~M) = 1;
end

function [I, M] = find_S(i, nrows, ~)
    I = i + 1;
    M = mod(i, nrows) ~= 0;
    I(~M) = 1;
end

function [I, M] = find_SW(i, nrows, ~)
    I = i - nrows + 1;
    M = mod(i, nrows) ~= 0 & I > 0;
    I(~M) = 1;
end

function [I, M] = find_W(i, nrows, ~)
    I = i - nrows;
    M = I > 0;
    I(~M) = 1;
end

function [I, M] = find_NW(i, nrows, ~)
    I = i - nrows - 1;
    M = I > 0;
    I(~M) = 1;
end


function [data, template_image, nlines, linewidth] = check_idata_inputsvars(data, template_image)

    assert(isnumeric(template_image), 'template_image must be numeric');
    assert(~isempty(template_image), 'template_image must be nonempty');
    assert(~any(isnan(template_image(:)) | isinf(template_image(:))), 'template_image cannot contains NaN or Inf values');
        
    if isa(data, 'function_handle')
        img = data(1);
        assert(isnumeric(img), 'data must be numeric');
        nlines = size(img, 1);
        linewidth = size(img, 2);
    else
        assert(isnumeric(data), 'data must be numeric');
        assert(~isempty(data), 'data must be nonempty');
        assert(~any(isnan(data(:)) | isinf(data(:))), 'data cannot contains NaN or Inf values');
        nlines = size(data, 1);
        linewidth = size(data, 2);
    end
    
    assert(size(template_image,1) * size(template_image,2) == numel(template_image), 'template_image must be a single image');
end

function settings = init_settings(settings, template, nlines, linewidth) 
    %initialize settings structure

    defaultsettings = default_alignment_settings();
    if isempty(settings)
        settings = defaultsettings;
    else
        fn = fieldnames(defaultsettings);
        for u = 1:numel(fn)
            if ~isfield(settings, fn{u})
                settings.(fn{u}) = defaultsettings.(fn{u});
            else
                %FIXME: should check that structure fields have valid values
            end
        end
    end
       
    %if no cropping specified assume no cropping
    if isempty(settings.cropping)
        settings.cropping = zeros(1,4);
    else
        assert(all(settings.cropping >= 0) && all(round(settings.cropping) == settings.cropping), 'cropping must be defined with positive integers');
    end
    
    croppeddatasize = [nlines linewidth] - [settings.cropping(3)+settings.cropping(4) settings.cropping(1)+settings.cropping(2)];
    assert(all(croppeddatasize > 0), 'no image pixels left');
    
    %if no subrec specified assume full template
    if isempty(settings.subrec)
        if  all(croppeddatasize == size(template)) 
            settings.subrec = [1 - settings.cropping(1), size(template, 2) + settings.cropping(2), 1 - settings.cropping(3), size(template, 1) + settings.cropping(4)];
        else
            %scan the whole template. this makes sense if the template is a mean or median or single frame
            settings.subrec = [1 size(template, 2) 1 size(template, 1)]; 
        end
    else 
        assert(settings.subrec(1) < settings.subrec(2) && settings.subrec(3) < settings.subrec(4), 'subrec is incorrectly described; use [left right bottom top]');
    end
            
    npixels = nlines * linewidth;
    
    %in normalized units so that the whole scan pattern has a duration of 1
    pixeldwelltime = 1 / npixels;
    %time when each pixel is scanned, same normalized units
    pixelscantime = pixeldwelltime / 2 + reshape(0:(npixels - 1), [linewidth nlines])' / npixels; 
    if settings.isbidirectional            
        pixelscantime(2:2:end, :) = fliplr(pixelscantime(2:2:end, :));
    end
    
    pixelscantime_croppeddata = cropimage(pixelscantime, settings.cropping);
    
    nlines = size(pixelscantime_croppeddata, 1);
    lines_per_knot =  nlines / (settings.nknots - 1);
    if lines_per_knot == round(lines_per_knot)           
        knotlines = 1:lines_per_knot:nlines;  % except for last knot
        t_knots = [min(pixelscantime(knotlines, :), [], 2)' - pixeldwelltime / 2, 1];
        t_knots(1) = 0;            
    else    
        %set up the knots so they cover the time equally for the data actually used, and move the first and last knots to cover the entire scan period.
        %this lets us equate the last knot of frame t with the first knot of frame t + 1
        t_knots = linspace(min(pixelscantime_croppeddata(:)) - pixeldwelltime / 2, max(pixelscantime_croppeddata(:)) + pixeldwelltime / 2, settings.nknots);
        t_knots(1) = 0;
        t_knots(end) = 1;
    end
    
    [~, prevknotind] = histc(pixelscantime_croppeddata, t_knots);
    assert(all(prevknotind(:) > 0) && all(prevknotind(:) < settings.nknots), 'error in setting up interpolation timing, pixel time is out of range with respect to interpolation knots');
    settings.prevknotind = prevknotind;
    
    settings.tfrac = (pixelscantime_croppeddata - t_knots(prevknotind)) ./ (t_knots(prevknotind + 1) - t_knots(prevknotind));
    
    if ~isfield(settings, 'basex') || ~isfield(settings, 'basey') || isempty(settings.basex) || isempty(settings.basey)
        [defaultbasex, defaultbasey] = meshgrid( ...
            linspace(0, 1, linewidth) * diff(settings.subrec(1:2)) + settings.subrec(1), ...
            linspace(0, 1, nlines) * diff(settings.subrec(3:4)) + settings.subrec(3)  );
        settings.basex = cropimage(defaultbasex, settings.cropping); 
        settings.basey = cropimage(defaultbasey, settings.cropping);
    else
        assert(all(size(basex) == croppeddatasize) || all(size(basey) == croppeddatasize), 'when basex and basey are given as inputs, they must match the size of the data after cropping');
    end    
end

function y = cropimage(x, cropvec)
    %y = cropimage(x, cropvec)
    %cropvec is [left right bottom top], nonnegative integers
    y = x(cropvec(3) + 1:end - cropvec(4), cropvec(1) + 1:end - cropvec(2), :);
end

function settings = default_alignment_settings()
    settings = orderfields(struct(...
        'move_thresh', 0.010,... %the converence threshold for changes in estimated displacements in pixels
        'corr_thresh',0.75,... %the minimum pearson's correlation value for a successful converence
        'max_iter',120,... %the maximum number of allowed red/green Lucas-Kanade iterations
        'nknots', 33,...%the number of parameters per frame
        'pregauss',0,... %amount of Gaussian filtering to apply to data before motion-correcting it
        'mednorm',0,...%should each input data frame be divided by its median?
        'min_blockpoints_ratio',0.3,...%fraction of pixels that must be available for a each block of time between two knots, in order for that block of data to be used
        'minblockfrac',0.25,... %fraction of blocks that must have been matched to data with at least one pixel for a successful alignment
        'haltcorr',0.995,... %the correlation value for which no further improvement will be attempted, even if the convergence criteria are not met
        'isbidirectional', false,... %was bidirectional scanning used (true or false), i.e. do even numbered scan lines go right to left instead of left to right
        'linesearch_reduction_multiplier',0.5,...%stepsize in linesearch
        'frames_per_set', 500,...%the maximum number of frames to motioncorrect on the gpu at the same time. if the last set has less then min_frames_per_set frames, the number of frames in the last two sets is averaged
        'min_frames_per_set',100,...%threshold in number of frames to rearrange sets (see frames_per_set)
        'cropping',[],... %how many pixels to ignore of the imaging data [left right bottom top]
        'subrec', [],...  %the rectangular region of template_image to which the raster scan was targeted. [left right bottom top]; note that subrec's values can be negative, when an undisplaced scan only partly overlaps with the template
        'ibs_inpaint',false,...
        'ibs_regfac',3e-3,...
        'ibs_template_pixel_aspect_ratio',1,...
        'ibs_version',2,... %algorithm used for reconstruction 1=fast but blurry images; 2=slow but sharp images
        'gpu_device_id',0,... %the id of the GPU to be used. possible ids are reported by nvidia-smi
        'gpu_frame_groupsize',2,... %the number of frames to combine for red/green updates. this should always be 2
        'gpu_correlation_increase_thresh',-1,... %threshold for change in pearson's correlation since the last update of the whole set at which to stop further improvement attemps
        'gpu_solver_id',2,... %id of the sparse solver used by displacement estimation; 1: cusparse solver 2:cusolver solver 3:Eigen solver
        'gpu_filter_id',1,... %id of the filter which is applied to the values after a gn-update; 0: disabled 1: smearing filter (slower+more accurate) 2:threshold filter (faster+less accurate(might fail to align)) 
        'gpu_filter_strength',0.8,... %filter strength [0..1]
        'gpu_filter_max_iterations',10,...%maximum number of iteration the filter will be activated
        'gpu_filter_correlation_thresh',0.01,... %threshold of pearson's correlation to deactivate the filter
        'gpu_forced_precision','D',... %precision type to use on the gpu; S: single precision D: double precision
        'gpu_flags','' ... %additional flags which can be passed for debugging purposes
    ));

    %possible flags for the gpu_flags parameter
    % T: time execution
    % t: time iterations / subfunctions
    % M: mask inactive groups
    % I: show parameter information
    % C: save correlations after each iteration
    % c: save all frame correlations
    % F: save call arguments to file (./VERSION_rom.bin)
    % W: show gateway conversion warnings/errors
    % w: treat gateway warnings as errors
    % G: show gpu memory usage
    % E: show stop criterion
    % R: enable 3-median filtering of parameters before linesearch
    % r: enable 3-median filtering of the resulting parameter values
    % f: notify when filter stage is deactivated
end



        

