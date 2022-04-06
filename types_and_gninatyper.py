'''
creates types and gninatypes files of the protein for input to CNN via libmolgrid
first argument is path to protein file
second argument is path to barycenters list file
'''
import molgrid
import struct
import numpy as np
import os
import sys
from pathlib import Path

def gninatype(file, gninatype_dir=None):
    # creates gninatype file for model input
    train_types=file.replace('.pdb','.types')
    with open(train_types, 'w') as f:
        f.write(file)
        
    atom_map=molgrid.FileMappedGninaTyper(str(Path(__file__).parent / 'gninamap'))
    dataloader=molgrid.ExampleProvider(atom_map,shuffle=False,default_batch_size=1)
    dataloader.populate(train_types)
    
    example=dataloader.next()
    coords=example.coord_sets[0].coords.tonumpy()
    types=example.coord_sets[0].type_index.tonumpy()
    types=np.int_(types) 

    if gninatype_dir is None:
        gninatype_file = file.replace('.pdb','.gninatypes')
    else:
        gninatype_file = os.path.join(gninatype_dir, os.path.basename(file).replace('.pdb','.gninatypes'))
    with open(gninatype_file,'wb') as fout:
        for i in range(coords.shape[0]):
            fout.write(struct.pack('fffi',coords[i][0],coords[i][1],coords[i][2],types[i]))
            
    os.remove(train_types)
    
    return gninatype_file


def gninatype2(gninamap_file, src_file, target_file):
    """
    Slightly more flexible input/output (Added by Dae) 
    """
    # creates gninatype file for model input
    temp_types_file = src_file.replace('.pdb','.types')
    with open(temp_types_file, 'w') as f:
        f.write(src_file)
        
    atom_map=molgrid.FileMappedGninaTyper(gninamap_file)
    dataloader=molgrid.ExampleProvider(atom_map,shuffle=False,default_batch_size=1)
    dataloader.populate(temp_types_file)
    
    example=dataloader.next()
    coords=example.coord_sets[0].coords.tonumpy()
    types=example.coord_sets[0].type_index.tonumpy()
    types=np.int_(types) 

    with open(target_file,'wb') as fout:
        for i in range(coords.shape[0]):
            fout.write(struct.pack('fffi',coords[i][0],coords[i][1],coords[i][2],types[i]))
            
    os.remove(temp_types_file)
    
    return target_file

def create_types(file,protein):
    # create types file for model predictions
    fout=open(file.replace('.txt','.types'),'w')
    fin =open(file,'r')
    for line in fin:
        fout.write(' '.join(line.split()) + ' ' + protein +'\n')
    return file.replace('.txt','.types')


if __name__ == '__main__':
    protein=gninatype(sys.argv[1])
    types=create_types(sys.argv[2],protein)
