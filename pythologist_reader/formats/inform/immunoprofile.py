from pythologist_reader.formats.inform.frame import CellFrameInForm, preliminary_threshold_read
from pythologist_reader.formats.inform.sets import CellSampleInForm, CellProjectInForm
from pythologist_reader.formats.inform.custom import CellFrameInFormLineArea, CellFrameInFormCustomMask
import os, re, sys
from tempfile import mkdtemp
from glob import glob
from shutil import copytree, copy, rmtree
import pandas as pd
from pythologist_image_utilities import read_tiff_stack, make_binary_image_array, binary_image_dilation
from uuid import uuid4


class CellProjectInFormImmunoProfile(CellProjectInForm):
    """
    Read an ImmunoProfile sample
    """
    def __init__(self,*argv,**kwargs):
        super().__init__(*argv,**kwargs)
        self.project_name = 'ImmunoProfile'
        return

    def create_cell_sample_class(self):
        return CellSampleInFormImmunoProfile()

    def add_sample_path(self,path,
                      sample_name=None,
                      export_names = ['FOXP3','PD1_PDL1'],
                      channel_abbreviations={
                                     'PD-L1 (Opal 520)':'PDL1',
                                     'Foxp3 (Opal 570)':'FOXP3',
                                     'PD-1 (Opal 620)':'PD1'},
                      verbose=False,
                      microns_per_pixel=0.496,
                      invasive_margin_width_microns=40,
                      invasive_margin_drawn_line_width_pixels=10,
                      skip_margin=False,
                      skip_segmentation_processing=False,
                      skip_all_regions=False,
                      deidentify=False,
                      **kwargs):
        """
        Read add a sample in as single project folder and add it to the CellProjectInFormImmunoProfile


        such as ``IP-99-A00001``:

        | IP-99-A00001/
        | └── INFORM_ANALYSIS
        |     ├── FOXP3
        |     ├── GIMP
        |     └── PD1_PDL1

        Args: 
            path (str): location of the project directory
            sample_name (str): name of the immunoprofile sample (default: rightmost directory in path), can be overridden by 'deidenitfy' set to True .. results in the uuid4 for the sample being used
            export_names (list): specify the names of the exports to read
            channel_abbreviations (dict): dictionary of shortcuts to translate to simpler channel names
            verbose (bool): if true print extra details
            microns_per_pixel (float): conversion factor
            invasive_margin_width_microns (int): size of invasive margin in microns
            invasive_margin_drawn_line_width_pixels (int): size of the line drawn for invasive margins in pixels
            skip_margin (bool): if false (default) read in margin line and define a margin acording to steps.  if true, only read a tumor and stroma.
            skip_segmentation_processing (bool): if false (default) read segementations, else skip to run faster
            deidentify (bool): if false (default) use sample names and frame names derived from the folders.  If true use the uuid4s.

        Returns:
            sample_id, sample_name (tuple) returns the uuid4 assigned as the sample_id, and the sample_name that were given to this sample that was added
        """

        if self.mode == 'r': raise ValueError("Error: cannot write to a path in read-only mode.")
        if sample_name is None: sample_name = os.path.split(path)[-1]


        # fix the margin width
        grow_margin_steps = int(invasive_margin_width_microns/microns_per_pixel-invasive_margin_drawn_line_width_pixels/2)
        if verbose: sys.stderr.write("To reach a margin width in each direction of "+str(invasive_margin_width_microns)+"um we will grow the line by "+str(grow_margin_steps)+" pixels\n")


        if microns_per_pixel is not None: self.microns_per_pixel = microns_per_pixel
        if verbose: sys.stderr.write("microns_per_pixel "+str(self.microns_per_pixel)+"\n")

        # read all terminal folders as sample_names unless there is none then the sample name is blank
        abspath = os.path.abspath(path)
        if not os.path.isdir(abspath): raise ValueError("Error project path must be a directory")
        if len(os.path.split(abspath)) < 2: raise ValueError("expecting an IP path structure")
        bpath1 = os.path.join(abspath,'INFORM_ANALYSIS')
        if not os.path.isdir(bpath1): raise ValueError("expecting an INFORM_ANLAYSIS directory as a child directory of IP path")


        #if autodectect_tumor:
        #    # Try to find out what the tumor is on this channel
        #    afiles = os.listdir(os.path.join(bpath1,export_names[0]))
        #    afiles = [x for x in afiles if re.search('_cell_seg_data.txt$',x)]
        #    if len(afiles) == 0: raise ValueError('expected some files in there')
        #    header = list(pd.read_csv(os.path.join(bpath1,export_names[0],afiles[0]),sep="\t").columns)
        #    cell = None
        #    for entry in header:
        #        m = re.match('Entire Cell (.* \('+autodectect_tumor+'\)) Mean \(Normalized Counts, Total Weighting\)',entry)
        #        if m: cell = m.group(1)
        #    if verbose and cell: sys.stderr.write("Detected the tumor channel as '"+str(cell)+"'\n")
        #    if cell: channel_abbreviations[cell] = 'TUMOR'
        #    #print(afile)


        if verbose: sys.stderr.write("Reading sample "+path+" for sample "+sample_name+"\n")

        # Read in one sample FOR this project
        cellsample = self.create_cell_sample_class()
        cellsample.read_path(path,sample_name=sample_name,
                                  channel_abbreviations=channel_abbreviations,
                                  verbose=verbose,
                                  require=True,
                                  require_score=True,
                                  skip_segmentation_processing=skip_segmentation_processing,
                                  export_names=export_names,
                                  deidentify=deidentify,
                                  steps = grow_margin_steps,
                                  )

        if deidentify: cellsample.sample_name = cellsample.id
        # Save the sample TO this project
        cellsample.to_hdf(self.h5path,location='samples/'+cellsample.id,mode='a')
        current = self.key
        if current is None:
            current = pd.DataFrame([{'sample_id':cellsample.id,
                                     'sample_name':cellsample.sample_name}])
            current.index.name = 'db_id'
        else:
            iteration = max(current.index)+1
            addition = pd.DataFrame([{'db_id':iteration,
                                      'sample_id':cellsample.id,
                                      'sample_name':cellsample.sample_name}]).set_index('db_id')
            current = pd.concat([current,addition])
        current.to_hdf(self.h5path,'info',mode='r+',complib='zlib',complevel=9,format='table')
        return cellsample.id, cellsample.sample_name


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
                            skip_margin=False,
                            skip_segmentation_processing=False,
                            skip_all_regions=False,
                            export_names=[],
                            deidentify=False):
        if len(export_names)==0: raise ValueError("You need to know the names of the export(s)")
        if sample_name is None: sample_name = path
        if not os.path.isdir(path):
            raise ValueError('Path input must be a directory')
        absdir = os.path.abspath(path)
        exportdir = os.path.join(absdir,'INFORM_ANALYSIS',export_names[0])
        files = os.listdir(exportdir)
        segs = [x for x in files if re.search('_cell_seg_data.txt$',x)]
        if len(segs) == 0: raise ValueError("There needs to be cell_seg_data in the folder.")
        frames = []
        if skip_margin and verbose: sys.stderr.write("FORCE SKIP ANY MARGIN FILES.. Tumor and Stroma Only\n")
        if skip_all_regions and verbose: sys.stderr.write("FORCE SKIP ALL REGION ANNOTATIONS .. Processed image will be annotated as a region 'Any'\n")
        for file in segs:
            m = re.match('(.*)cell_seg_data.txt$',file)
            score = os.path.join(exportdir,m.group(1)+'score_data.txt')
            #summary = os.path.join(path,m.group(1)+'cell_seg_data_summary.txt')
            parent = os.path.split(exportdir)[0]
            #print(path)
            binary_seg_maps = os.path.join(exportdir,m.group(1)+'binary_seg_maps.tif')
            component_image = os.path.join(exportdir,m.group(1)+'component_data.tif')
            tfile = os.path.join(exportdir,m.group(1)+'tissue_seg_data.txt')
            tumor = os.path.join(parent,'GIMP',m.group(1)+'Tumor.tif')
            margin = os.path.join(parent,'GIMP',m.group(1)+'Invasive_Margin.tif')
            tissue_seg_data = tfile if os.path.exists(tfile) else None
            frame = m.group(1).rstrip('_')
            data = os.path.join(exportdir,file)
            if not os.path.exists(score):
                    raise ValueError('Missing score file '+score)
            if verbose: sys.stderr.write('Acquiring frame '+data+"\n")
            cid = None
            if os.path.exists(margin) and not skip_margin and not skip_all_regions:
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
                             require=require,
                             skip_segmentation_processing=skip_segmentation_processing)
                #print(cid)
                update_with_other_scores(cid,parent,m.group(1),export_names[1:])
                if verbose: sys.stderr.write("growing margin by "+str(steps)+" steps\n")
                if not skip_all_regions: cid.set_line_area(margin,tumor,steps=steps,verbose=verbose)
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
                         require_score=require_score,
                         skip_segmentation_processing=skip_segmentation_processing)
                #print(cid)
                # Must update the score file before refactoring regions
                update_with_other_scores(cid,parent,m.group(1),export_names[1:])
                stroma_name = 'Stroma-No-Margin'
                if os.path.exists(margin) and skip_margin: stroma_name = 'Stroma-Ignore-Margin'
                if not skip_all_regions: cid.set_area(tumor,'Tumor',stroma_name,verbose=verbose)

            if deidentify: cid.frame_name = cid.id
            frame_id = cid.id
            self._frames[frame_id]=cid
            frames.append({'frame_id':frame_id,'frame_name':frame,'frame_path':absdir})
            if verbose: sys.stderr.write("finished tumor and stroma and margin\n")
        self._key = pd.DataFrame(frames)
        self._key.index.name = 'db_id'
        self.sample_name = sample_name

def update_with_other_scores(frame, parent, file_prefix, alt_folders):
    # Now lets look for additional scores for this frame
    for altfolder in alt_folders:
        # see if there is an approrpriate score in this
        altpath = os.path.join(parent,altfolder,file_prefix+'score_data.txt')
        if not os.path.exists(altpath): 
                    if verbose: sys.stderr.write("WARNING: Missing a score file in the alternate folder "+str(altpath)+"\n")
                    continue
        # If we are still here we have a score file
        # This part is a little hacky .. we are going to bring a function from an CellFrameInForm just so we can use its "preliminary_threshold_read" function
        altscore = preliminary_threshold_read(altpath, frame.get_data('measurement_statistics'), 
                                                       frame.get_data('measurement_features'), 
                                                       frame.get_data('measurement_channels'), 
                                                       frame.get_data('regions')).reset_index().copy()
        current_max = max(frame.get_data('thresholds').index)
        altscore['gate_index'] = altscore['gate_index'].apply(lambda x: x+current_max+1)
        newscore = pd.concat([frame.get_data('thresholds').reset_index(),altscore],sort=True).set_index('gate_index')
        frame.set_data('thresholds',newscore)
    return


