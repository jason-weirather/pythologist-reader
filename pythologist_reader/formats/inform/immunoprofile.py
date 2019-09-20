from pythologist_reader.formats.inform.frame import CellFrameInForm
from pythologist_reader.formats.inform.sets import CellSampleInForm, CellProjectInForm
from pythologist_reader.formats.inform.custom import CellFrameInFormLineArea, CellFrameInFormCustomMask
import os, re, sys
from tempfile import mkdtemp
from glob import glob
from shutil import copytree, copy, rmtree
import pandas as pd
from pythologist_image_utilities import read_tiff_stack, make_binary_image_array, binary_image_dilation
from uuid import uuid4

def read_InFormImmunoProfileV1(path,
                                  save_FOXP3_intermediate_h5=None,
                                  save_PD1_PDL1_intermediate_h5=None,
                                  channel_abbreviations={'Foxp3 (Opal 570)':'FOXP3',
                                                         'PD-1 (Opal 620)':'PD1',
                                                         'PD-L1 (Opal 520)':'PDL1',
                                                         'CD8 (Opal 480)':'CD8',
                                                         'Cytokeratin (Opal 690)':'CYTOKERATIN'},
                                  grow_margin_steps=40,
                                  microns_per_pixel=0.496,
                                  project_name = 'ImmunoProfileV1',
                                  project_id_is_project_name=True,
                                  skip_margin=False,
                                  auto_fix_phenotypes=True,
                                  verbose=False,
                                  tempdir=None):
    """
    Read the InForm Exports from ImmunoProfile and merge them into a single CellDataFrame.

    Args: 
        path (str): location of the ImmunoProfile sample or folder of samples
        save_FOXP3_intermediate_h5 (str): path to save the FOXP3 export images as h5.  Keep this one if you want to tie the CellDataFrame to the images.
        save_PD1_PDL1_intermediate_h5 (str): path to save the PD1_PDL1 export images as h5. Probably do not save this unless you are trying to debug a failed import.
        channel_abbreviations (dict): convert stain names to abbreviations
        grow_margin_steps (int): number of pixels to grow the margin
        microns_per_pixel (float): conversion factor for pixels to microns
        project_name (str): name of the project
        verbose (bool): if true print extra details
        skip_margin (bool): if false (default) read in margin line and define a margin acording to steps.  if true, only read a tumor and stroma.
        auto_fix_phenotypes (bool): if true (default) automatically try to fill in any missing phenotypes with zero-values.  This most commonly happens when there are no CD8's on an image and thus the image is not phenotyped for them.
        project_id_is_project_name (bool): if true (default) make the project_id be the same as your project_name.  This will make concatonating sample dataframes simpler.
    Returns:
        Pass (CellDataFrame): Cells that merged properly
        Fail (CellDataFrame): Cells that failed to merge properly (non-zero indicates a QC issue for either missing data or unmatched segmentation)
    """
    if verbose: sys.stderr.write("=========Reading FOXP3 Export=========\n")
    if tempdir is not None and not os.path.exists(tempdir): os.makedirs(tempdir)
    mytempdir = mkdtemp(dir=tempdir)
    temp1 = os.path.join(mytempdir,'FOXP3.h5')
    if save_FOXP3_intermediate_h5 is not None: temp1 = save_FOXP3_intermediate_h5
    cpi1 = CellProjectInFormImmunoProfile(temp1,mode='w')
    cpi1.read_path(path,'FOXP3',project_name=project_name,
                                verbose=verbose,
                                channel_abbreviations=channel_abbreviations,
                                steps=grow_margin_steps,
                                microns_per_pixel=microns_per_pixel,
                                skip_margin=skip_margin)
    if verbose: sys.stderr.write("\n\n=========Reading PD1 PDL1 Export=========\n")
    temp2 = os.path.join(mytempdir,'PD1_PDL1.h5')    
    if save_PD1_PDL1_intermediate_h5 is not None: temp2 = save_PD1_PDL1_intermediate_h5
    cpi2 = CellProjectInFormImmunoProfile(temp2,mode='w')
    cpi2.read_path(path,'PD1_PDL1',project_name=project_name,
                                   verbose=verbose,
                                   channel_abbreviations=channel_abbreviations,
                                   steps=grow_margin_steps,
                                   microns_per_pixel=microns_per_pixel,
                                   skip_margin=skip_margin)
    if tempdir is None: rmtree(mytempdir)
    cdf1 = cpi1.cdf
    cdf2 = cpi2.cdf
    if verbose: sys.stderr.write("\n\n=======Merging FOXP3 with PD1 PDL1 Export=======\n")
    p,f = cdf1.merge_scores(cdf2,on=['project_name','sample_name','frame_name','cell_index'])

    if auto_fix_phenotypes and verbose:
        sys.stderr.write("\n\n=======Quick QC 1=========\n")
        qc = p.qc()
        qc.run_tests()
        for test in qc._tests:
            sys.stderr.write("---------\n")
            sys.stderr.write("  "+str(test.name)+"\n")
            sys.stderr.write("  "+str(test.result)+"\n")
            sys.stderr.write("  "+str(test.about)+"\n")
            if test.total is not None: sys.stderr.write('Issue count: '+str(test.count)+'/'+str(test.total))

    if auto_fix_phenotypes and not p.is_uniform(): 
        if verbose: sys.stderr.write("FIXING non-uniform with zero-fill.\n")
        p = p.zero_fill_missing_phenotypes()

    did_fix = False
    if did_fix and verbose:
        p = p.zero_fill_missing_phenotypes()
        did_fix = True
        sys.stderr.write("\n\n=======Quick QC 2 (POST FIX)=========\n")
        qc = p.qc()
        qc.run_tests()
        for test in qc._tests:
            sys.stderr.write("---------\n")
            sys.stderr.write("  "+str(test.name)+"\n")
            sys.stderr.write("  "+str(test.result)+"\n")
            sys.stderr.write("  "+str(test.about)+"\n")
            if test.total is not None: sys.stderr.write('Issue count: '+str(test.count)+'/'+str(test.total))
    if p.shape[0] > 0 and project_id_is_project_name:
        p['project_name'] = p['project_id']
    if f.shape[0] > 0 and project_id_is_project_name:
        f['project_name'] = f['project_id']
    return p,f

class CellProjectInFormImmunoProfile(CellProjectInForm):
    """
    Read in an ImmunoProfile sample that could have either a Tumor mask alone, or a Tumor mask and a hand drawn margin,
    this will read the two projects into a cell project.  This will only read in one InForm export at a time.

    Accessed via ``read_path`` with the additonal parameters
    """
    def create_cell_sample_class(self):
        return CellSampleInFormImmunoProfile()
    def read_path(self,path,
                      export_name=None,
                      project_name=None,
                      channel_abbreviations=None,
                      verbose=False,
                      require=True,
                      require_score=True,
                      microns_per_pixel=None,
                      steps=40,
                      skip_margin=False,
                      **kwargs):
        """
        Read in the project folder

        This can either be 1) a project folder or 2) a folder of folders

        
        1) 


        ```
        IP-MYCASE1
           |- FOXP3
              |-frame1
              |-frame2
           |-PD1_PDL1
             |-frame1
             |-frame2
           |-GIMP
             |-frame1_Tumor.tif
             |-frame2_Tumor.tif
             |-frame2_Invasive_Margin.tif
        ```

        or 2)

        ```
        MYFOLDER
           |-IP-MYCASE1
           |-IP-MYCASE2
           |-IP-MYCASE3
        ```


        Args: 
            path (str): location of the project directory
            export_name (str): specify the name of the export to read (required)
            project_name (str): name of the project
            channel_abbreviations (dict): dictionary of shortcuts to translate to simpler channel names
            verbose (bool): if true print extra details
            require (bool): if true (default), require that channel componenet image be present
            require_score (bool): if true (default), require there be a score file in the data
            microns_per_pixel (float): conversion factor
            steps (int): number of pixels to grow the margin
            skip_margin (bool): if false (default) read in margin line and define a margin acording to steps.  if true, only read a tumor and stroma.
        """
        if export_name is None: raise ValueError("specify the name of the panel to read")
        if project_name is not None: self.project_name = project_name
        if microns_per_pixel is not None: self.microns_per_pixel = microns_per_pixel
        if verbose: sys.stderr.write("microns_per_pixel "+str(self.microns_per_pixel)+"\n")
        if self.mode == 'r': raise ValueError("Error: cannot write to a path in read-only mode.")
        # read all terminal folders as sample_names unless there is none then the sample name is blank
        abspath = os.path.abspath(path)
        if not os.path.isdir(abspath): raise ValueError("Error project path must be a directory")
        sample_dirs = set()
        for root, dirs, files in os.walk(abspath):
            if len(dirs) > 0: continue
            if os.path.split(root)[-1] != export_name: continue
            if len(os.path.split(root)[-1]) < 2: raise ValueError("expecting an IP path structure")
            bpath1 = os.path.split(root)[0]
            if os.path.split(bpath1)[-1] != 'INFORM_ANALYSIS': continue
            if len(os.path.split(bpath1)) < 2: raise ValueError("expecting an IP path structure")
            sample_dirs.add(root)
        for s in sample_dirs:
            #sname = None
            #if sample_name_index is None: sname = s
            sname  = s.split(os.sep)[-3]
            sid = self.add_sample_path(s,sample_name=sname,
                                         channel_abbreviations=channel_abbreviations,
                                         verbose=verbose,require=require,
                                         require_score=require_score,
                                         steps=steps,
                                         skip_margin=skip_margin,
                                         **kwargs)
            if verbose: sys.stderr.write("Added sample "+sid+"\n")

class CellSampleInFormImmunoProfile(CellSampleInForm):
    def create_cell_frame_class(self):
        return CellFrameInFormLineArea() # this will be called when we read the HDF
    def create_cell_frame_class_line_area(self):
        return CellFrameInFormLineArea()
    def create_cell_frame_class_custom_mask(self):
        return CellFrameInFormCustomMask()
    def read_path(self,path,sample_name=None,
                            channel_abbreviations=None,
                            verbose=False,
                            require=True,
                            require_score=True,
                            steps=76,
                            skip_margin=False):
        if sample_name is None: sample_name = path
        if not os.path.isdir(path):
            raise ValueError('Path input must be a directory')
        absdir = os.path.abspath(path)
        z = 0
        files = os.listdir(path)
        z += 1
        segs = [x for x in files if re.search('_cell_seg_data.txt$',x)]
        if len(segs) == 0: raise ValueError("There needs to be cell_seg_data in the folder.")
        frames = []
        if skip_margin and verbose: sys.stderr.write("FORCE SKIP ANY MARGIN FILES.. Tumor and Stroma Only\n")
        for file in segs:
            m = re.match('(.*)cell_seg_data.txt$',file)
            score = os.path.join(path,m.group(1)+'score_data.txt')
            #summary = os.path.join(path,m.group(1)+'cell_seg_data_summary.txt')
            parent = os.path.split(path)[0]
            #print(path)
            binary_seg_maps = os.path.join(path,m.group(1)+'binary_seg_maps.tif')
            component_image = os.path.join(path,m.group(1)+'component_data.tif')
            tfile = os.path.join(path,m.group(1)+'tissue_seg_data.txt')
            tumor = os.path.join(parent,'GIMP',m.group(1)+'Tumor.tif')
            margin = os.path.join(parent,'GIMP',m.group(1)+'Invasive_Margin.tif')
            tissue_seg_data = tfile if os.path.exists(tfile) else None
            frame = m.group(1).rstrip('_')
            data = os.path.join(path,file)
            if not os.path.exists(score):
                    raise ValueError('Missing score file '+score)
            if verbose: sys.stderr.write('Acquiring frame '+data+"\n")
            cid = None
            if os.path.exists(margin) and not skip_margin:
                if verbose: sys.stderr.write("LINE AREA TYPE\n")
                cid = self.create_cell_frame_class_line_area()
                cid.read_raw(frame_name = frame,
                             cell_seg_data_file=data,
                             score_data_file=score,
                             tissue_seg_data_file=tissue_seg_data,
                             binary_seg_image_file=binary_seg_maps,
                             component_image_file=component_image,
                             channel_abbreviations=channel_abbreviations,
                             verbose=verbose,
                             require=require)
                cid.set_line_area(margin,tumor,steps=steps,verbose=verbose)
            else:
                if verbose: sys.stderr.write("TUMOR MASK ONLY TYPE\n")
                cid = self.create_cell_frame_class_custom_mask()
                cid.read_raw(frame_name = frame,
                         cell_seg_data_file=data,
                         score_data_file=score,
                         tissue_seg_data_file=tissue_seg_data,
                         binary_seg_image_file=binary_seg_maps,
                         component_image_file=component_image,
                         channel_abbreviations=channel_abbreviations,
                         verbose=verbose,
                         require=require,
                         require_score=require_score)
                stroma_name = 'Stroma-No-Margin'
                if os.path.exists(margin) and skip_margin: stroma_name = 'Stroma-Ignore-Margin'
                cid.set_area(tumor,'Tumor',stroma_name,verbose=verbose)
            frame_id = cid.id
            self._frames[frame_id]=cid
            frames.append({'frame_id':frame_id,'frame_name':frame,'frame_path':absdir})
            if verbose: sys.stderr.write("finished tumor and stroma and margin\n")
        self._key = pd.DataFrame(frames)
        self._key.index.name = 'db_id'
        self.sample_name = sample_name