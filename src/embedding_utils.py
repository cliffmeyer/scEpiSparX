import h5py
import numpy as np
import os
import torch


def get_out_of_vocab_embeddings(h5_filename):
    with h5py.File(h5_filename,'r') as fp:
        null_embeddings =  1 * torch.tensor(np.ones(fp['embeddings'].shape[1]))
    return null_embeddings


def get_embedding_dim(h5_filename):
    with h5py.File(h5_filename,'r') as fp:
        num_embeddings = fp['embeddings'].shape[0]
        embedding_dim = fp['embeddings'].shape[1]
    return num_embeddings, embedding_dim


def get_embeddings(h5_filename):
    with h5py.File(h5_filename, 'r') as fp:
        print('HDF5 keys:', list(fp.keys())) 
        embeddings = torch.tensor( fp['embeddings'][:] )
    return embeddings


def extract_chrom_start_end(region):
    """
    Get chrom, start and end from genomic interval string. 
    args:
        - region_str: eg "chr1:1000-2000" 
    returns:
        chrom (str), start (int), end (int)
    """
    try:
        chrom, pos = region.split(':')
        start, end = map(int, pos.split('-'))
    except:
        return 'UNKNOWN', 0, 0
    return chrom, start, end


def region_strings_to_tuples(region_strings,chr_to_idx={}):
    """
    Convert format of genomic intervals from string format 
    to tuples. eg ['chr1:1000-2000',...] to [(0,1000,2000),...] or [('chr1',1000,2000)]
    args:
        - region_strings (list of str): genomic intervals in string format
        - chr_to_idx (dict): 
            if empty chr is represented at string, 
            otherwise lookup integer index of chromosomes eg {'chr':0,...}
    returns:
        (list of tuples) eg. [('chr1',1000,2000), ... ] or [(0,1000,2000),...]
    """ 
    # Process regions to extract chrom, start, end
    region_data = [extract_chrom_start_end(r) for r in region_strings]
    # Filter out 'UNKNOWN' regions
    region_data = [r for r in region_data if r[0] != 'UNKNOWN']
    # Convert chroms to indices
    if len(chr_to_idx):
        region_tuples = [ (chr_to_idx[chrom],start,end) for chrom,start,end in region_data ]
    else:
        region_tuples = [ (chrom,start,end) for chrom,start,end in region_data ]
    return region_tuples


def get_regions(h5_filename):
    # TODO check
    """
    args:
        - h5_filename: (str) path to HDF file
    returns:
        - regions: (torch tensor) contains chr (str), start (int), end (int) info for each region 
    """
    with h5py.File(h5_filename, 'r') as fp:
        _regions = fp['regions'][:]
        
        if _regions.dtype.kind == 'O': # string format 
            regions_strings = [r.decode('utf-8') for r in _regions]
            regions = region_strings_to_tuples(regions_strings, chr_to_idx={})
        elif _regions.dtype.kind == 'i':  # integer format
            idx_to_chr = {i: v.decode('utf-8') for i,v in enumerate(fp['chroms'][:])}
            regions = [(idx_to_chr[idx], start, end) for idx, start, end in _regions]
        else:
            raise ValueError 

        # TODO check tuple converts to tensor?
        regions = torch.tensor( regions )
    return regions


def make_hdf5_region_index(h5_filename):
    """
    read in chromosome index and region index from hdf5.
    HDF5 file has the following structure:
         dataset *chroms*: array of chromosome names as strings eg. chr1, chr2, ..
         dataset *regions*: array of genomic intervals chrom_index, start, end, index is derived from chroms 
         dateset *embeddings*: array of embeddings in same order as regions
    args:
        - h5_filename (str): name of HDF5 file containing genomic interval embeddings

    return dict keys are (chr (str),start (int), end (int)) and vals index for looking up embeddings

    NOTE: chrom format is chr1, chr2, etc in lookup table, therefore is easily interpreted
    """
    _CHROM,_START,_END=0,1,2 

    if not os.path.exists(h5_filename):
        return {}

    with h5py.File(h5_filename, 'r') as fp:

        chrom_idx = { i:chrom.decode('utf-8') for i,chrom in enumerate(fp['chroms']) }
        regions = fp['regions']
        region_lookup = { ( chrom_idx[r[_CHROM]], r[_START], r[_END] ):idx for idx,r in enumerate(regions) }

    return region_lookup


def make_region_to_embeddings(h5_filename):
    """
    read in chromosome index, region index and embeddings from HDF5 file and return
    regions to embeddings map.
    args:
        - h5_filename: file containing genomic region embeddings
    returns:
        - dict: { region: embedding, } 
              where region is in format ( chr (str), start (int), end (int) )
              and embedding is torch tensor
    """
    region_to_index = make_hdf5_region_index(h5_filename)
    embeddings = get_embeddings(h5_filename)
    region_to_embeddings = { k:embeddings[idx] for k,idx in region_to_index.items() }
    return region_to_embeddings


def get_hdf5_row_indices(nearby_regions, region_lookup, NOT_FOUND=None):
    """
    args:
        - nearby_regions (list): regions to look up 
        - region_lookup (dict): map from region to int
        - NOT_FOUND: value to fill in if a region is not in the lookup, if NOT_FOUND is None regions are filtered out
    """
    if NOT_FOUND is not None:
        indices = [region_lookup.get(region, NOT_FOUND) for region in nearby_regions]
    else:
        indices = [region_lookup.get(region, NOT_FOUND) for region in nearby_regions]
        indices = [elem for elem in indices if elem]
    return np.array(indices)


def compose_embeddings(h5_fps=[], null_embeddings = [], region_lookups=[], nearby_regions=[], weights=[], max_regions=15):
    """
    Compose embeddings from multiple HDF5 files
    args:
        - h5_fps (list): open h5 File objects
        - null_embeddings (list): list of null embeddings, for regions with missing embeddings by HDF5 file
        - region_lookups (list of dicts): each list corresponds with an HDF5 file mapping region to row in embedding array
        - nearby_regions (list): list of regions, region format (chr_index,start,end) 
        - weights (list): weight for each region
        - max_regions (int): limit on number of regions to include (by weight) 
    returns
        - valid_regions: (list) of regions in same format as input arg nearby_regions 
        - valid_weights: (list) or weights for each valid region
        - embeddings: (pytorch tensor len(valid_regions) x embedding_dim torch.float32 ) 
    """
    NOT_FOUND = -1

    # Get region indices for each HDF5 file
    region_idx = [get_hdf5_row_indices(nearby_regions, region_lookups[i], NOT_FOUND) for i in range(len(h5_fps))]
    unknown_region_idx = [get_hdf5_row_indices([('UNKNOWN',0,0)], region_lookups[i], NOT_FOUND) for i in range(len(h5_fps))]

    #print( 'nearby regions', nearby_regions )
    #print( 'region idx', region_idx )
    #print( 'unknown  region idx', unknown_region_idx )

    # Find regions with data and filter weights        
    region_idx_array = np.array(region_idx)  # n_h5 x n_nearby_regions
    valid_idx = np.where(np.any(region_idx_array != NOT_FOUND, axis=0))[0]
    valid_weights = np.array(weights)[valid_idx]

    # Limit to max_regions
    top_i = np.argsort(valid_weights)[-max_regions:][::-1]
    top_sorted_i = top_i[np.argsort(valid_weights[top_i])[::-1]]
    valid_idx = valid_idx[top_sorted_i]
    valid_weights = valid_weights[top_sorted_i]

    # Sort indices
    region_idx_array = region_idx_array[:, valid_idx] # TODO check

    # check there are regions
    if valid_idx.shape[0] == 0:
        return None, None, None

    _embeddings_list = []

    # Get embeddings for each HDF5 and fill in missing embeddings
    for i_h5, h5_fp in enumerate(h5_fps):
        
        #print( 'Embedding shape:', h5_fp['embeddings'].shape )

        embedding_dim = h5_fp['embeddings'].shape[1]
        _valid_region_idx = region_idx_array[i_h5] # check
        _unknown_region_idx = unknown_region_idx[i_h5][0]

        _mask = _valid_region_idx != NOT_FOUND
        _embeddings = torch.zeros( (len(_valid_region_idx), embedding_dim), dtype=torch.float32 )

        #print( 'valid region idx __', _valid_region_idx[_mask] )

        #print( 'mask', _mask )
        if _mask.any():

            _z = list(zip(_valid_region_idx[_mask], range(len(_valid_region_idx[_mask]))))
            _sorted = sorted(_z)
            _sorted_region_idx, _sorted_idx = zip(*_sorted)

            sorted_embeddings = torch.tensor(h5_fp['embeddings'][_sorted_region_idx,:], dtype=torch.float32)

            _embeddings[_mask] = sorted_embeddings[np.argsort(_sorted_idx)]
        
        #_embeddings[~_mask] = torch.tensor(null_embeddings[i_h5], dtype=torch.float32)  # Handle NOT_FOUND
        _embeddings[~_mask] = null_embeddings[i_h5].clone().detach().float()  # Handle NOT_FOUND

        _embeddings_list.append(_embeddings)
    
    # Concatenate embeddings
    embeddings = torch.cat(_embeddings_list, dim=1 )
    valid_regions = [nearby_regions[i] for i in valid_idx] 
    #return  valid_regions, valid_weights, torch.tensor( embeddings, dtype=torch.float32 )
    return valid_regions, valid_weights, embeddings.clone().detach().float()

