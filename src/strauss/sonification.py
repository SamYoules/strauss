""" :obj:`sonification`: generate sonification, combining submodules.

This Submodule handles the combining of all the constituent
subroutines into  a single :obj:`sonification` object that can then
render and output/save the resultant sonification. This handles
feeding of information between :obj:`strauss` modules, including
taking the :obj:`sources` mapping, applying any musical constraints
from :obj:`score` running the :obj:`generators` to make sound and
combining them into the output channels for the overall spatialised
sonificiation.

Todo:
  * Delegate more musical process to the :obj:`score` module
"""

from .stream import Stream
from .channels import audio_channels
import contextlib
import io
from .utilities import const_or_evo, nested_dict_idx_reassign, apply_fades, rescale_values, NoSoundDevice
from .tts_caption import render_caption, get_ttsMode, default_tts_voice
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import subprocess as sp
import wavio as wav
import IPython.display as ipd
from IPython.display import display
from scipy.io import wavfile
import warnings
import tempfile
from pathlib import Path
import ffmpeg
try:
    import sounddevice as sd
except (OSError, ModuleNotFoundError) as sderr:
    sd = NoSoundDevice(sderr)
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = list

# ----------------------------------------------------------------------
# Globals
# ----------------------------------------------------------------------

# fix audio sample rate
SAMPRATE = 48000

# maximum absolute sample value for audio peak normalisation
MAXSAMP = (2**31)-1

# supported sequence types
seq_types = ['animation',
             'image',
             'text',
             'blank',
             'clip']

defaults = {'fps': '30', 		# frames per second
            'crf': '18',		# ffmpeg quality level
            'invert_colours': True,	# dark on light is true
            'transition_time': '30',	# sequence transition time [frames]
            'breathing_time': '6',	# minimum time between separate sounds [frames]
            #'dimensions': '4200x2100',
            'background_video': './example_media/starfield.mov',
            #'background_video': '/Users/jamestrayford/Downloads/stockvideo_01171.mov',
            'dimensions': '3840x2160',   # video dimensions (4k by standard)
            'orientation': 'vertical',
            'transition_type': 'fade',
            'slide_min_margin': '300',
            'slide_key_black': '0',
            'clip_override_duration': '1'
            }

# ----------------------------------------------------------------------
# Useful dictionaries
# ----------------------------------------------------------------------

orient = {'vertical': '3840x2160',
          'horizontal': '2160x3840'}

res = {'4k': (3840, 2160),
       '1080p': (1920, 1080),
       '720p' : (1280, 720)}

class Sonification:
    """Representing the overall sonification

    This class combines the data sources, musical score constraints
    and generator together to generate and render the ultimate
    sonification for saving or playing in the :obj:`jupyter-notebook`
    environment 


    Todo:
      * Support custom audio setups here too.
    """
    def __init__(self, score, sources, generator, audio_setup='stereo',
                 caption=None, samprate=48000, declick_time=0.03,
                 ttsmodel=default_tts_voice):
        """
        Args:
         score (:class:`~strauss.score.Score`): Sonification :obj:`Score`
    	  object 
         sources (:class:`~strauss.sources.Source`): Sonification
    	  :obj:`Sources` child object (:class:`~strauss.sources.Events`
    	  or :class:`~strauss.sources.Objects`)  
         generator (:class:`~strauss.generator.Generator`): Sonification
    	  :obj:`Generator` child object
    	  (:class:`~strauss.generator.Synthesizer` or
    	  :class:`~strauss.generator.Sampler`)
         audio_setup (:obj:`str`) The requested audio setup preset to
    	  pass to :class:`~strauss.channels.audio_channels`
         samprate (:obj:`int`) Integer sample rate in samples per second
          (Hz), typically :obj:`44100` or :obj:`48000` for most audio
    	  applications
         declick_time (:obj:`float`) duration of start and end fades applied
          on save and dispolay to remove audible clicks from sample
          discontinuity
         ttsmodel (:obj:`str` or :obj:`PosixPath`) file path to the
          text-to-speech model used for captions. 
        """
        
        # sampling rate in Hz
        self.samprate = samprate
        
        # tts model name
        self.ttsmodel = ttsmodel

        # fade duration to de-click audio
        self.declick_time = declick_time
        
        # caption
        self.caption = caption
        
        # sonification owns an instance of the Score
        self.score = score
        
        # sonification owns an instance of the Sources
        self.sources = sources

        # sonification owns an instance of the Generator
        self.generator = generator
        
        # set up the audio channel routing for the sonification
        self.channels = audio_channels(setup=audio_setup)

        # check Generator and Sonification sampling rates match...
        if self.samprate != self.generator.samprate:
            # if not, revert to Generator sampling rate.
            warnings.warn("warning: global and generator sampling rates disagree, " \
            f"reverting to generator value of {self.generator.samprate} Hz")
            self.samprate = self.generator.samprate
        
        # ...and the corresponding Stream objects 
        self.out_channels = {}
        for c in range(self.channels.Nmics):
            self.out_channels[str(c)] = Stream(self.score.length, self.samprate)

    def render(self, downsamp=1):
        """Render the sonification.
        
        Generates the sonification by running the  Synthesizer
        :func:`~strauss.generator.Synthesizer.play` or Sampler
        :func:`~strauss.generator.Sampler.play` functions, and
        combining these into the output channel streams using any
        spatialisation for the specified
        :class:`~strauss.channels.audio_channels`. 

        Args:
          downsamp (optional, :obj:`int`): Optionally downsample
           sources for multi-source sonifications for a quicker test
           render by some integer factor.
        """

        # first determine if time is provided, if not assume all start at zero
        # and last the duration of sonification

        if "time" not in self.sources.mapping:
            self.sources.mapping['time'] = [0.] * self.sources.n_sources
            self.sources.mapping['note_length'] = [self.score.length] * self.sources.n_sources
            
        # index each chord
        cbin = np.digitize(self.sources.mapping['time'], self.score.fracbins, 0)
        cbin = np.clip(cbin-1, 0, self.score.nchords-1)

        # pitch rank of each source divided by the number of sources
        pitchfrac = np.empty_like(self.sources.mapping['pitch'])
        if self.score.pitch_binning == 'adaptive':
            pitchfrac[np.argsort(self.sources.mapping['pitch'])] = np.arange(self.sources.n_sources)/self.sources.n_sources
        elif self.score.pitch_binning == 'uniform':
            pitchfrac = np.clip(self.sources.mapping['pitch'], 0, 9.999999e-1)
            
        # get some relevant numbers before iterating through sources
        Nsamp = self.out_channels['0'].values.size
        lastsamp = Nsamp - 1
        Nchan = len(self.out_channels.keys())
        indices = range(0,self.sources.n_sources, downsamp)

        for source in tqdm(indices):

            # index note properties
            t = self.sources.mapping['time'][source]
            tsamp = int(Nsamp * t)
            chord = self.score.note_sequence[cbin[source]]
            nints = self.score.nintervals[cbin[source]]
            pitch = pitchfrac[source]
            note = chord[int(pitch * nints)]

            # make dictionary for feeding to play function with each notes properties
            sourcemap = {}
            # for k in self.sources.mapping.keys():
            #     sourcemap[k] = self.soures.mapping[k][source]
            nested_dict_idx_reassign(self.sources.mapping, sourcemap, source)

            sourcemap['note'] = note

            # run generator to play each note
            sstream = self.generator.play(sourcemap)
            playlen = sstream.values.size
            if 'phi' in sourcemap:
                azi     = const_or_evo(sourcemap['phi'], sstream.sampfracs) * 2 * np.pi
            elif 'azimuth' in sourcemap:
                azi     = const_or_evo(sourcemap['azimuth'], sstream.sampfracs) * 2 * np.pi
            else:
                azi     = const_or_evo(self.generator.preset['azimuth'], sstream.sampfracs) * 2 * np.pi
            if 'theta' in sourcemap:
                polar   = const_or_evo(sourcemap['theta'], sstream.sampfracs) * np.pi
            elif 'polar' in sourcemap:
                polar   = const_or_evo(sourcemap['polar'], sstream.sampfracs) * np.pi                
            else:
                polar   = const_or_evo(self.generator.preset['polar'], sstream.sampfracs) * np.pi

            # compute sample indices for truncating notes overshooting sonification length
            trunc_note = min(playlen, lastsamp-tsamp)
            trunc_soni   = trunc_note + tsamp

            # spatialise audio by computing relative volume in each speaker
            for i in range(Nchan):
                panenv = self.channels.mics[i].antenna(azi,polar)
                self.out_channels[str(i)].values[tsamp:trunc_soni] += (sstream.values*panenv)[:trunc_note]

        # produce mono audio of caption, if one is provided
        if str(self.caption or '').strip():
            ttsMode = get_ttsMode() # determine if using coqui-ai or pyttsx3

            # use a temporary directory to ensure caption file cleanup
            with tempfile.TemporaryDirectory() as cdir:
                cpath = Path(cdir, 'caption.wav')
                render_caption(self.caption, self.samprate,
                               self.ttsmodel, str(cpath))
                rate_in, wavobj = wavfile.read(cpath)
                wavobj = np.array(wavobj)
            # Set up the Stream objects for TTS
            self.caption_channels = {}
            caption_norm = wavobj.max()
            for c in range(Nchan):
                self.caption_channels[str(c)] = Stream(wavobj.shape[0], self.samprate, ltype='samples')
                
                # place caption straight ahead spatially
                panenv = self.channels.mics[c].antenna(0, 0.5*np.pi)
                
                cnorm = abs(self.out_channels[str(c)].values).max()/caption_norm
                self.caption_channels[str(c)].values += (wavobj*cnorm*panenv)
        else:
            self.caption_channels = {}
            for c in range(Nchan):
                self.caption_channels[str(c)] = Stream(0, self.samprate) 


    def add_ticks(self, increment, duration=0.04, tick_vol=0.25):
        # TODO this should probably use a dedicated generator...

        # add tick volume to Sonification object
        self.tick_vol = tick_vol
        
        tick_samples = 2*(np.random.random(self.out_channels['0'].values.shape)-0.5)
        k = 'time'
        if k not in self.sources.lims.keys():
            k = 'time_evo'
            if k not in self.sources.lims.keys():
                raise Exception("""
                Sonification doesn't have a time base! only sonifications with a 'time'
                or 'time_evo' mapping can have time increment ticks...
                """)
        inc = self.score.length*rescale_values(self.sources.lims[k][0]+increment,
                                               self.sources.lims[k],
                                               self.sources.plims[k])
        self.t_per_inc = np.linspace(0, self.score.length/inc, tick_samples.shape[0])
        self.tdur_per_inc = inc/duration
        tickenv = np.clip(1/self.tdur_per_inc - self.t_per_inc%1, 0, np.inf)
        tickenv /= tickenv.max()
        tick_samples = tick_samples*tickenv
        Nchan = len(self.out_channels.keys())
        self.tick_channels = {}
        for i in range(Nchan):
            panenv = self.channels.mics[i].antenna(0, 0.5*np.pi)
            self.tick_channels[str(i)] = Stream(tick_samples.size, self.samprate, ltype='samples')
            self.tick_channels[str(i)].values += tick_samples * panenv


    def save_stereo(self, fname, master_volume=1.):
        """ Save stereo or mono sonifications
        
        Can use this function to save :obj:`"stereo"` or :obj:`"mono"`
        sonifications while avoiding ffmpeg processing.

        Args:
          fname (:obj:`str`) Filename or filepath
          master_volume (:obj:`float`) Amplitude of the largest volume
            peak, from 0-1

        Todo:
          * Support :obj:`master_volume` in decibels
        """

        if len(self.out_channels) > 2:
            print("Warning: sonification has > 2 channels, only first 2 will be used. See 'save_combined' method.")


        # first pass - find max amplitude value to normalise output
        # and concatenate channels to list
        vmax = 0.
        channels = []
        for c in range(min(len(self.out_channels), 2)):
            vmax = max(
                abs(self.out_channels[str(c)].values.max()),
                abs(self.out_channels[str(c)].values.min()),
                vmax
            ) / master_volume
            
            # combine caption + sonification streams at display time
            channel_values = np.concatenate([self.out_channels[str(c)].values,
                                self.caption_channels[str(c)].values])   
            
            channels.append(channel_values)

        wav.write(fname,
                  np.column_stack(channels),
                  self.samprate, 
                  scale = (-vmax,vmax),
                  sampwidth=3)

        print("Saved.")


    def save_combined(self, fname, ffmpeg_output=False, master_volume=1.):
        """ Save render as a combined multi-channel wav file 
        
        Can use this function to save sonification of any audio_setup,
        using ffmpeg processing, and unscrampling to the correct
        channel order.

        Args:
          fname (:obj:`str`) Filename or filepath
          ffmpeg_output (:obj:`bool`) If True, print :obj:`ffmpeg`
            output to screen 
          master_volume (:obj:`float`) Amplitude of the largest volume
            peak, from 0-1
        """
        # setup list to house wav stream data 
        inputs = [None]*len(self.out_channels)

        # first pass - find max amplitude value to normalise output
        vmax = 0.
        for c in range(len(self.out_channels)):
            vmax = max(
                abs(self.out_channels[str(c)].values.max()),
                abs(self.out_channels[str(c)].values.min()),
                vmax
            ) / master_volume
            
        print("Creating temporary .wav files...")

        # combine caption + sonification streams at display time
        for c in range(len(self.out_channels)):
            tempfname = Path('.', f'.TEMP_{c}.wav')
            self.out_channels[str(c)].values += self.caption_channels[str(c)].values
            wav.write(tempfname, 
                      self.out_channels[str(c)].values,
                      self.samprate, 
                      scale = (-vmax,vmax),
                      sampwidth=3)
            inputs[self.channels.forder[c]] = ff.input(tempfname)
            
        print("Joining temporary .wav files...")
        (
            ff.filter(inputs, 'join', inputs=len(inputs), channel_layout=self.channels.setup)
            .output(fname)
            .overwrite_output()
            .run(quiet=~ffmpeg_output)
        )
        
        print("Cleaning up...")
        for c in range(len(self.out_channels)):
            Path('.', f'.TEMP_{c}.wav').unlink()
            
        print("Saved.")

    def save(self, fname, master_volume=1., embed_caption=True):
        """ Save render as a combined multi-channel wav file 
        
        Can use this function to save sonification of any audio_setup
        to a file. This first creates a 32-bit depth WAV using
        `scipy.io.wavfile`. If fname has a non-WAV extension, it then attempts
        conversion via ffmpeg, provided ffmpeg is available.

        formats

        Args:
          fname (:obj:`str`) Filename or filepath
          master_volume (:obj:`float`) Amplitude of the largest volume
            peak, from 0-1
          embed_caption (:obj:`bool`) Whether or not to embed caption
            at the start of the output audio

        Todo:
          * Raise `scipy` issue if common 24-bit WAV can be supported
        """

        channels = []
        vmax = 0.

        has_ticks = hasattr(self, 'tick_channels')

        # first pass - find max amplitude value to normalise output
        for c in range(len(self.out_channels)):

            channel_values = np.concatenate(int(embed_caption)*[self.caption_channels[str(c)].values,]+
                                            [apply_fades(self.out_channels[str(c)].values,
                                                         self.out_channels['0'].samprate,
                                                         fdur=self.declick_time)])
            channels.append(channel_values)
            vmax = max(
                abs(channels[c].max()),
                abs(channels[c].min()),
                vmax
            ) * 1.05

        # normalisation for conversion to int32 bitdepth wav
        norm = master_volume * (pow(2, 31)-1) / vmax

        # setup array to house wav stream data 
        chans = np.zeros((channels[0].size, len(channels)), dtype="int32")
        
        # normalise and collect channels into a list
        for c in range(len(self.out_channels)):
            signal = channels[c]*norm
            if has_ticks:
                # add the ticks
                signal += self.tick_channels[str(c)].values*norm*self.tick_vol
            chans[:,c] = (signal).astype("int32")

        # finally combine and write out file. first check extension
        fsplit = str(fname).split('.')
        if len(fsplit) < 2:
            warnings.warn('No file extension in provided fname. Assuming WAV...')
        ext = fsplit[-1].lower()
        if ext != 'wav':
            # check we can use ffmpeg binary
            try:
                sp.run(['ffmpeg','-h'],capture_output=1, check=1)
            except FileNotFoundError as e:
                raise FileNotFoundError(f"""
                'ffmpeg' doesn't appear to be available in the local environment.
                This may need to be installed manually. To install ffmpeg visit
                https://www.ffmpeg.org/download.html.
                {str(e)}
                """)
            # Use NamedTemporaryFile with delete=False to avoid file locking issues on Windows
            # when passing the file to a subprocess
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp_name = tmp.name

            try:
                # now first write the wav to a temporary file
                wavfile.write(tmp_name, self.samprate, chans)
                try:
                    # try (naive) convert with ffmpeg
                    sp.run(['ffmpeg', '-y', '-i', f'{tmp_name}', f'{fname}'],
                           capture_output=1, check=1)
                except sp.CalledProcessError as e:
                    # if ffmpeg can't do it for whatever reason, raise
                    raise Exception(f"""
                    'ffmpeg' failed to convert '.wav' to '.{ext}' succesfully:
                    {str(e)}
                    {e.stderr}""")
            finally:
                if os.path.exists(tmp_name):
                    os.remove(tmp_name)
        else:
            wavfile.write(fname, self.samprate, chans)

        print(f"Saved {fname}")

        
    def notebook_display(self, show_waveform=True):
        """ plot the waveforms and embed player in the notebook

        Show waveforms and embed an audio player in the python
        notebook for direct playback. the notebook player only
        supports up to stereo, so if more than two channels, only the
        first two are used as left and right.
        """

        time = self.out_channels['0'].samples / self.out_channels['0'].samprate

        has_ticks = hasattr(self, 'tick_channels')
        channels = []
        fig = plt.figure(figsize=(18,12))
        vmax = 0.
        
        # combine caption + sonification streams at display time
        for c in range(len(self.out_channels)):
            # apply fades at display time
            channel_values = np.concatenate([self.caption_channels[str(c)].values,
                                             apply_fades(self.out_channels[str(c)].values,
                                                         self.out_channels['0'].samprate,
                                                         fdur=self.declick_time)])
            channels.append(channel_values)
            vmax = max(
                abs(channels[c].max()),
                abs(channels[c].min()),
                vmax
            ) * 1.05
        
        if show_waveform:
            for i in range(len(self.out_channels)):
                plt.plot(time[::20], self.out_channels[str(i)].values[::20]+2*i*vmax, label=self.channels.labels[i])
            plt.xlabel('Time (s)')
            plt.ylabel('Relative Amplitude')
            plt.legend(frameon=False, loc=5)
            plt.xlim(-time[-1]*0.05,time[-1]*1.2)
            for s in plt.gca().spines.values():
                s.set_visible(False)
                plt.gca().get_yaxis().set_visible(False)
            plt.show()
        
        if len(self.channels.labels) == 1:             
            # we have used 48000 Hz everywhere above as standard, but to quickly hear the sonification sped up / slowed down,
            # you can modify the 'rate' argument below (e.g. multiply by 0.5 for half speed, by 2 for double speed, etc)
            outfmt = np.column_stack(channels*2).T / vmax
        else:
            outfmt = np.column_stack(channels[:2]).T / vmax
        if len(self.channels.labels) > 2:
            print("Warning: for more than two channels, only first two channels are mapped to L and R, respectively.")
        if has_ticks:
            # add the ticks
            for c in range(outfmt.shape[0]):
                outfmt[c] += self.tick_channels['0'].values*self.tick_vol / vmax
        display(ipd.Audio(outfmt,rate=self.out_channels['0'].samprate, autoplay=False))
        
    def hear(self):
        """ Play audio directly to the sound device, for command-line playback.

        If available, use the ``sounddevice`` module to stream the sonification to
        the sound device directly (speakers, headphones, etc.) via the underlying
        ``PortAudio`` C-library. if unavaialable, raise error.

        Todo:
          * Add more options to control the streamed audio
        """

        channels = []
        vmax = 0.
        
        # combine caption + sonification streams at display time
        for c in range(len(self.out_channels)):
            channel_values = np.concatenate([self.caption_channels[str(c)].values,
                                             self.out_channels[str(c)].values])   
            channels.append(channel_values)
            vmax = max(
                abs(channels[c].max()),
                abs(channels[c].min()),
                vmax
            ) * 1.05
                
        if len(self.channels.labels) == 1:             
            # we have used 48000 Hz everywhere above as standard, but to quickly hear the sonification sped up / slowed down,
            # you can modify the 'rate' argument below (e.g. multiply by 0.5 for half speed, by 2 for double speed, etc)
            outfmt = np.column_stack(channels*2)/vmax
        else:
            outfmt = np.column_stack(channels[:2])/vmax

        dur = int(np.round(outfmt.shape[0]/self.out_channels['0'].samprate))
        playback_msg = f"Playing Sonification ({dur} s): "
        print(playback_msg)
        try:
            sd.play(outfmt,self.out_channels['0'].samprate,blocking=1)
        except OSError as error: 
            print(error) 
            print("The Sonification.hear() function requires the PortAudio C-library. This may be missing from your system or \n"
                  "unsupported in this context. This should be installed by pip on Windows and OSx automatically with the \n "
                  "sounddevice library, but on Linux you may need to install manually using e.g.:\n"
                  "\t 'sudo apt-get install libportaudio2.'\n")

    def _make_seamless(self, overlap_dur=0.05):
        """ Make a seamlessly looping audio signal.

        Audio signal is made seamless by cross-fading end of signal back into start
        over a duration (in seconds) defined by ``overlap_dur``

        Args:
          overlap_dur (:obj:`float`): cross-fade duration in seconds.        
        """
        self.loop_channels = {}
        buffsize = int(overlap_dur*self.samprate)
        ramp = np.linspace(0,1, buffsize+1)
        for c in range(len(self.out_channels)):
            self.loop_channels[str(c)] = Stream(self.out_channels[str(c)].values.size - buffsize,
                                                self.samprate, ltype='samples')
            self.loop_channels[str(c)].values = self.out_channels[str(c)].values[:-buffsize]
            self.loop_channels[str(c)].values[:buffsize] *= ramp[:-1]
            self.loop_channels[str(c)].values[:buffsize] += ramp[::-1][:-1] * self.out_channels[str(c)].values[-buffsize:]
            

#================================================================================
#===== helper functions =========================================================
#================================================================================

@contextlib.contextmanager
def suppress_output():
    save_stdout = sys.stdout
    sys.stdout = io.BytesIO()
    yield
    sys.stdout = save_stdout


def generate_caption(caption, path, notebook=True):
    #mode = tts.ttsMode
    #voices = tts.getVoices(True)
    #if mode == 'coqui-tts':
    #    tts = TTS(model_name='tts_models/en/jenny/jenny', progress_bar=False, gpu=False)
    #elif mode == 'pyttsx3':
    #    tts = TTS(model_name={'voice':v.id, 'rate': 217}, progress_bar=False, gpu=False)
    #tts.tts_to_file(text=caption, file_path=path)


    #tts.render_caption(caption, samprate=48000, model, path)
    if TTS is None:
        raise ImportError("TTS module not found. Please install 'coqui-tts' to use animation captions.")

    tts = TTS(model_name='tts_models/en/jenny/jenny', progress_bar=False, gpu=False)
    tts.tts_to_file(text=caption, file_path=path)

def prep_caption(caption):
    if caption:
        caption = caption.strip('.!?¿¡')
        sents = caption.split('.')
        return '.'.join(sents[:-1] + [sents[-1]+'.'])
    return None

def force_stereo(audio_file, do_resample=False):
    sound = wav.read(audio_file)

    data = sound.data

    if do_resample:
        print(int(sound.rate), SAMPRATE)
        data = resample(int(sound.rate), SAMPRATE, data)

    if data.shape[1] == 1:
        audio = np.column_stack([data, data])
    else:
        audio = data[:,:2]
    return audio.astype(float)

def house_audio(audio, spf, fpad=0):
    fpad = int(fpad)
    spf = int(spf)
    halfpad = (spf*fpad // 2)
    zarr = np.zeros(((-int(-(audio.shape[0]/spf) // 1) + fpad) *
                     spf,2))
    # print(audio.shape, spf, fpad, halfpad, zarr.shape)
    zarr[halfpad:audio.shape[0]+halfpad] = audio
    return zarr


def render_transition(fromfile, tofile, toseq):
    # get transition frames
    # ffmpeg -i inputfile.mkv -vf "select=eq(n\,0)" -q:v 3 output_image.jpg
    inframe = '/'.join(fromfile.split('/')[:-1] + ['from.png'])
    outframe = '/'.join(tofile.split('/')[:-1] + ['to.png'])
    sp.check_call(["ffmpeg", '-y',
                     "-sseof", '-0.2',
                     '-i', fromfile,
                     '-update', '1',
                     inframe],
                    stdout=sp.DEVNULL, stderr=sp.STDOUT)

    sp.check_call(["ffmpeg", '-y',
                     '-i', tofile,
                     '-vf', "select=eq(n\\,0)",
                     outframe],
                    stdout=sp.DEVNULL, stderr=sp.STDOUT)

    # make transition video
    tdur = int(toseq.pars['transition_time'])/int(toseq.pars['fps'])
    toff = 0
    transfile = '/'.join(tofile.split('/')[:-1] + ["transin.mp4"])
    tvidpars = ['-r', toseq.pars['fps'],
                '-loop', '1',
                '-t', str(tdur)]

    print(f"\t Transition into sequence {toseq.name}...")
    sp.check_call(["ffmpeg", '-y',
                     *tvidpars,
                     '-i', inframe,
                     *tvidpars,
                     '-i', outframe,
                     '-filter_complex',
                     f"[0][1]xfade=transition={toseq.pars['transition_type']}:duration={tdur}:offset={toff}",#format=yuv420p",
                     '-bsf:v', 'h264_metadata=sample_aspect_ratio=1/1',
                     '-c:v', 'libx264',
                     '-crf', toseq.pars["crf"],
                     '-c:a', 'libx264',
                     transfile],
                    stdout=sp.DEVNULL, stderr=sp.STDOUT)

def prepare_clip(seq, infile, outtype):
    dims = seq.pars['dimensions'].split('x')
    margin = int(seq.pars['slide_min_margin'])
    filts = []
    filts.append(f"scale=w={int(dims[0])-margin}:h={int(dims[1])-margin}:force_original_aspect_ratio=1")
    if not int(seq.pars['slide_key_black']):
        # this subtle brightening ensures all pixels are outside keyed range (above absolute black)
        filts.append(f"eq=brightness=0.04")
    filts.append(f"pad={dims[0]}:{dims[1]}:(ow-iw)/2:(oh-ih)/2")

    # extract audio
    sp.check_call(["ffmpeg", '-y',
                   '-i', infile,
                   f"{seq.path}/{outtype}.wav"],
                  stdout=sp.DEVNULL, stderr=sp.STDOUT)

    # reencode video
    cmd = ["ffmpeg", '-y',
                   '-i', infile,
                   "-vf", ",".join(filts),
                   '-r', seq.pars["fps"],
                   '-c:v', 'libx264',
                   '-crf', seq.pars["crf"],
                   '-c:a', 'libx264',
                   f"{seq.path}/{outtype}.mp4"]

    # print (' '.join(cmd))
    sp.check_call(cmd, stdout=sp.DEVNULL, stderr=sp.STDOUT)

def generate_slide_video(seq, still, outtype, time=None, nframes=None):
    dims = seq.pars['dimensions'].split('x')
    margin = int(seq.pars['slide_min_margin'])
    filts = []
    filts.append(f"scale=w={int(dims[0])-margin}:h={int(dims[1])-margin}:force_original_aspect_ratio=1")
    if not int(seq.pars['slide_key_black']):
        # this subtle brightening ensures all pixels are outside keyed range (above absolute black)
        filts.append(f"eq=brightness=0.04")
    filts.append(f"pad={dims[0]}:{dims[1]}:(ow-iw)/2:(oh-ih)/2")

    if time:
        dur = ['-t', str(time)]
    if nframes:
        dur = ['-frames', str(nframes)]

    cmd = ["ffmpeg", '-y',
                   "-loop", "1",
                   '-i', still,
                   "-vf", ",".join(filts),
                   '-r', seq.pars["fps"],
                   dur[0], dur[1],
                   '-c:v', 'libx264',
                   '-crf', seq.pars["crf"],
                   '-c:a', 'libx264',
                   f"{seq.path}/{outtype}.mp4"]

    # print (' '.join(cmd))
    sp.check_call(cmd, stdout=sp.DEVNULL, stderr=sp.STDOUT)

# ----------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------

class Animate:

    def __init__(self, topdir, pars={}):
        if topdir.exists() and any(topdir.iterdir()):
            warnings.warn(f"{topdir} is not empty, instead name "
                          "an empty directory, or a new one.")
        self.topdir = topdir

        # handle parameters
        self.pars = defaults.copy()
        for k in pars.keys():
            self.pars[k] = pars[k]
        self.pars['spf'] = SAMPRATE/int(self.pars['fps'])
        self.spf = SAMPRATE/int(self.pars['fps'])

        if self.spf % 1:
            Exception("non integer samples-per-frame value, please use a standard video"\
                      "fps (30,25,20) and audio sample rate (48000, 44100)")

        self.sequences = {}
        self.frames = {}
        self.seqlist = []
        self.seqdx = 0

        # stereo audio ramps to prevent dropouts, 30ms hard coded
        self.aramplen = int(0.03*SAMPRATE)
        self.arampin = np.column_stack([np.linspace(0,1, self.aramplen)]*2)
        self.arampout = self.arampin[::-1]

        # audio padding for breathing time and transitions
        self.apadbreath = np.column_stack([np.zeros(int(self.pars['spf'] * float(self.pars['breathing_time'])))]*2)
        self.apadtrans = np.column_stack([np.zeros(int(self.pars['spf'] * float(self.pars['transition_time'])))]*2)
        self.halfbsamps = self.apadtrans.size // 2

    def register(self, name, dark_mode=True, sonification=None, pre_caption='', post_caption='', stype='animation', infile=None, pars={}):
        #if ((duration * (int(self.pars['fps']) + SAMPRATE)) % 1) and (stype != 'clip'):
        #    Exception(f"Duration {duration}s gives a non-integer number of frames and/or audio samples," \
        #              f"please retry, for example with an integer number of seconds e.g. ({int(np.ceil(duration))}s)")
        duration = sonification.score.length
        if ((duration * (int(self.pars['fps']) + SAMPRATE)) % 1) and (stype != 'clip'):
            duration = math.ceil(duration)

        inpars = self.pars.copy()
        for k in pars.keys():
            inpars[k] = pars[k]
        self.sequences[name] = Sequence(name, duration=duration,
                                        topdir=self.topdir, index=self.seqdx,
                                        sonification=sonification,
                                        pre_caption=pre_caption,
                                        post_caption=post_caption,
                                        pars=inpars, stype=stype,
                                        invert_colours=dark_mode, infile=infile)
        self.seqlist.append(name)
        self.frames[name] = self.sequences[name].frame
        self.seqdx += 1

    def render(self):

        master = []
        flist = []
        fromfile = None

        print("First, process sequences.\n")
        print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")

        # first, render sequences
        for i in range(len(self.seqlist)):

            name = self.seqlist[i]
            seq = self.sequences[name]
            seqvid = f'{seq.path}/{seq.name}.mp4'

            print(f"Sequence {i}: \t {name}")
            subfs = []

            # render steps, per sequence
            seq.render_frames()
            seq.render_caption()
            seq.render_sonification()
            seq.render_caption_stills()

            # compile sonification audio
            prepath = str(Path(seq.path)/'pre.wav')
            if seq.pre:
                audio = force_stereo(prepath)
                audio *= MAXSAMP / abs(audio).max()
                audio = house_audio(audio, self.pars['spf'], self.pars['breathing_time'])
                master.append(audio)
                # append pre vid sequence
                subfs.append(prepath[:-3]+'mp4')

            # compile sonification audio
            if seq.sonification:
                audio = force_stereo(str(seq.audiofile))

                # ramp audio
                audio[:self.aramplen] *= self.arampin
                audio[-self.aramplen:] *= self.arampout

                audio *= MAXSAMP / abs(audio).max()
                audio = house_audio(audio, self.pars['spf'])

            elif seq.stype == 'clip':
                print('in')
                audio = force_stereo(str(seq.audiofile), do_resample=1)
                audio *= MAXSAMP / abs(audio).max()
                audio = house_audio(audio, self.pars['spf'])

            else:
                # stereo silence
                audio = np.zeros((int(seq.duration*SAMPRATE), 2))

            # append vid sequence
            subfs.append(seqvid)

            # append audio
            master.append(audio)

            postpath = str(Path(seq.path)/'post.wav')
            if seq.post:
                audio = force_stereo(postpath)
                audio *= MAXSAMP / abs(audio).max()
                audio = house_audio(audio, self.pars['spf'], self.pars['breathing_time'])
                master.append(audio)
                # append post vid sequence
                subfs.append(postpath[:-3]+'mp4')

            # pad for transition out
            master.append(self.apadtrans.copy())

            if fromfile:
                render_transition(fromfile, seqvid, seq)
                subfs = [seq.path+'/transin.mp4'] + subfs

            flist += subfs
            fromfile = flist[-1]


        print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n")
        print("Now, compile material.")
        # remove final transition
        # for a in master:
        #     print('duration: ', a.shape[0]/int(SAMPRATE))
        outsamps = np.vstack(master[:-1]).astype('int32')

        self.master_wav = str(Path(self.topdir)/'master.wav')
        self.master_mp3 = str(Path(self.topdir)/'master.mp3')
        self.combined = str(Path(self.topdir)/'combo.mp4')
        self.concat_file = str(Path(self.topdir)/'concat_files.txt')
        self.final = str(Path(self.topdir)/'final.mp4')

        wavfile.write(self.master_wav, SAMPRATE, outsamps)

        with open(self.concat_file, "w") as cfiles:
            for f in flist:
                cfiles.write(f"file '{f}'\n")

        # concatenate files!
        print(f"Concatenate files...")
        sp.check_call(["ffmpeg", '-y',
                         "-f", 'concat',
                         '-safe', '0',
                         '-i', self.concat_file,
                         '-c', 'copy',
                         self.combined],
                        stdout=sp.DEVNULL, stderr=sp.STDOUT)

        # make and convert master audio track (run python script)
        print(f"Convert audio...")
        sp.check_call(["ffmpeg", '-y',
                         "-i", self.master_wav,
                         '-vn', '-ar', str(SAMPRATE),
                         '-ac', '2', '-b:a', '192k',
                         self.master_mp3],
                        stdout=sp.DEVNULL, stderr=sp.STDOUT)

        bgfile = glob.glob(self.pars['background_video'])

        if bgfile:
            print(f"Combine sequences and chroma-key background video...")
            tempfile = str(Path(self.topdir)/'overlay.mp4')
            sp.check_call(["ffmpeg", '-y',
                             '-stream_loop', '-1',
                             '-i', bgfile[0],
                             '-i', self.combined,
                             '-filter_complex',
                             '[1:v]colorkey=0x000000:0.01:0.01[ckout];[0:v][ckout]overlay=(W-w)/2:(H-h)/2:shortest=1[out]',
                             '-map', '[out]', tempfile],
                            stdout=sp.DEVNULL, stderr=sp.STDOUT)
            self.combined = tempfile

        else:
            print(f"Background video file '{self.pars['background_video']}' not found, skipping...")

        print(f"Dubbing final video...")
        sp.check_call(["ffmpeg", '-y',
                         '-i', self.combined,
                         '-itsoffset', '0.047', # TODO: investigate why extra 47ms padding is needed to sync?
                         '-i', self.master_mp3,
                         #'-map', '0:v', '-map', '1:a', '-c:a', 'libvorbis',
                         '-map', '0:v', '-map', '1:a',
                         '-c:v', 'copy', '-shortest',
                         self.final],
                        stdout=sp.DEVNULL, stderr=sp.STDOUT)

        print("Done!")

class Sequence:
    def __init__(self, name, duration, sonification=None, topdir='', index=None, stype='animation', custom_path=None,
                 invert_colours=True, pars=defaults, pre_caption='', post_caption='', infile=None):
        self.name = name
        if stype == "clip":
            dur = sp.run(["ffprobe", "-v", "error", "-show_entries",
                          "format=duration", "-of",
                          "default=noprint_wrappers=1:nokey=1", infile],
                         stdout=sp.PIPE, stderr=sp.STDOUT)
            self.duration = float(dur.stdout)
        else:
            self.duration = duration
        self.pars = pars
        self.infile = infile

        self.index = index
        self.subs = {}
        self.stype = stype
        self.path = Path(topdir) / name
        self.frame = Path(self.path)  / f"frame_{index:05d}.png"
        self.nframes = int(np.ceil(int(self.pars['fps']) * self.duration))
        self.sonification = sonification

        if sonification and (duration is not sonification.score.length):
            Exception(f"Provided sonification length ({sonification.score.length}s) != sequence duration ({duration}s)")

        self.audiofile = Path(self.path) / f"{name}.wav"
        self.pars = pars
        self._torender_flags = {'pre': True, 'post': True,
                                'frames': True, 'sonification': bool(sonification)}

        self.lastdx = None
        self.length = None

        if pre_caption:
            self.pre = prep_caption(pre_caption)
        else:
            self.pre = ''

        if post_caption:
            self.post = prep_caption(post_caption)
        else:
            self.post = ''

        print(f"making {self.path}")
        Path(self.path).mkdir(parents=True, exist_ok=True)

    def caption(self, pre='', post=''):
        if self.pre != pre:
            self.pre = prep_caption(pre)
            self._torender_flags['pre'] = True
        if self.post != post:
            self.post = prep_caption(post)
            self._torender_flags['post'] = True

    def render_caption(self, notebook=True):
        print(f"\t Rendering {self.name} captions:")
        if self.pre:
            fpath = Path(self.path) / 'pre.wav'
            print(f'\t\t pre-caption: "{self.pre}" to {fpath}')
            # with suppress_output():
            with contextlib.redirect_stdout(None):
                generate_caption(self.pre, fpath, notebook)
            self._torender_flags['pre'] = False
        if self.post:
            fpath = Path(self.path) / 'post.wav'
            print(f'\t\t post-caption: "{self.post}" to {fpath}')
            # with suppress_output():
            with contextlib.redirect_stdout(None):
                generate_caption(self.post, fpath, notebook)
            self._torender_flags['post'] = False

    def render_sonification(self):
        if self._torender_flags['sonification'] and self.sonification:
            # check it's been rendered
            if not self.sonification.out_channels['0'].values.any():
                print(f"\t Rendering Sonification for {self.name} sequence...")
                with contextlib.redirect_stdout(None):
                    self.sonification.render()
            with contextlib.redirect_stdout(None):
                self.sonification.save(self.audiofile)

    def render_frames(self):
        if self._torender_flags['frames']:
            inv = ""
            if self.pars["invert_colours"]:
                inv = "-vf negate"
            outfile = str(Path(self.path)/f"{self.name}.mp4")

            # store the number of frames
            # nframes = len(glob.glob(f'{self.path}/frames*'))
            # self.lastdx = nframes-1
            # self.duration = nframes / int(self.pars["fps"])

            # make video from frames...
            print(f"\t Render video for sequence {self.name} {self.stype}...")

            if self.stype == 'animation':
                print(str(Path(self.path) / f'{self.name}.png'))
                # TODO: decide how failure-permitted subprocesses should be run?
                #sp.check_call(["ffmpeg", '-y',
                #                 '-r', self.pars["fps"],
                #                 '-i', str(Path(self.path) / f'{self.name}.png'),
                #                 '-c:v', 'libx264',
                #                 '-crf', self.pars["crf"]] +
                #                inv.split() + [outfile],
                #                stdout=sp.DEVNULL, stderr=sp.STDOUT)
                sp.run(['ffmpeg', '-y',
                        '-r', self.pars["fps"],
                        '-i', str(Path(self.path) / f'frame_%05d.png'),
                        '-c:v', 'libx264',
                        '-c:a', 'libx264',
                        '-crf', self.pars["crf"]] +
                        inv.split() + [outfile],
                        stdout=sp.DEVNULL, stderr=sp.STDOUT,
                        check=False)

            elif self.stype == 'slide':
                # make slide sequence
                generate_slide_video(self, self.infile, self.name, time=self.duration)

            elif self.stype == 'clip':
                # make slide sequence
                prepare_clip(self, self.infile, self.name)

            else:
                sp.check_call(['ffmpeg', '-y',
                               '-f', 'lavfi',
                               '-i', f'color=c=black:s={self.pars["dimensions"]}',
                               '-frames', str(self.duration * int(self.pars['fps'])),
                               '-r', self.pars["fps"],
                               '-c:v', 'libx264',
                               '-crf', self.pars["crf"],
                               '-c:a', 'libx264',
                               outfile],
                               stdout=sp.DEVNULL, stderr=sp.STDOUT)


            # frames rendered for now...
            self._torender_flags['frames'] = False

    def render_caption_stills(self):
        print(f"\t Rendering {self.name} caption stills...")
        # iterate through existing captions
        video = str(Path(self.path)/f"{self.name}.mp4")
        pos = 0
        ctype = ["pre", "post"]
        for c in [self.pre, self.post]:
            if c:
                print(f"\t\t Making {ctype[pos]}-caption still for {self.name}...")
                clen = wav.read(str(Path(self.path) / f'{ctype[pos]}.wav')).data.shape[0]
                nframes = clen / self.pars['spf']
                nframes = -int(-nframes // 1) + int(self.pars['breathing_time'])
                if (self.stype == 'animation') and (glob.glob(str(self.frame).format(index=0))):
                    fnum = pos*(int(self.pars['fps'])*self.duration - 1)
                    frame = str(self.frame).format(index=int(fnum))
                    if self.pars["invert_colours"]:
                        inv = "-vf negate"

                    # make still sequence
                    sp.check_call(['ffmpeg', '-y',
                                   '-loop', '1',
                                   '-i', frame,
                                   '-r', self.pars["fps"],
                                   '-frames', str(nframes),
                                   '-c:v', 'libx264',
                                   '-c:a', 'libx264',
                                   '-crf', self.pars["crf"]] +
                                   inv.split() +
                                   [str(Path(self.path)/f"{ctype[pos]}.mp4")],
                                   stdout=sp.DEVNULL, stderr=sp.STDOUT)

                elif self.stype == 'slide':
                    generate_slide_video(self, self.infile, ctype[pos], nframes=nframes)

                elif self.stype == 'clip':
                    if pos:
                        frame = str(Path(self.path)/f'pre.png')
                        sp.check_call(['ffmpeg', '-y',
                                       '-sseof', '-0.1',
                                       '-i', Path(self.path)/f'{self.name}.mp4',
                                       '-update', '1',
                                       frame],
                                       stdout=sp.DEVNULL, stderr=sp.STDOUT)
                    else:
                        frame = str(Path(self.path)/f'post.png')
                        sp.check_call(['ffmpeg', '-y',
                                       '-i', str(Path(self.path)/f'{self.name}.mp4'),
                                       '-vf', "select=eq(n\\,0)",
                                       frame],
                                       stdout=sp.DEVNULL, stderr=sp.STDOUT)
                   # make still sequence
                    sp.check_call(['ffmpeg', '-y',
                                   '-loop', '1',
                                   '-i', frame,
                                   '-r', self.pars["fps"],
                                   '-frames', str(nframes),
                                   '-c:v', 'libx264',
                                   '-crf', self.pars["crf"],
                                   '-c:a', 'libx264',
                                  str(Path(self.path)/f"{ctype[pos]}.mp4")],
                                   stdout=sp.DEVNULL, stderr=sp.STDOUT)

                else:
                    # blank video
                    sp.check_call(['ffmpeg', '-y',
                                   '-f', 'lavfi',
                                   '-i', f'color=c=black:s={self.pars["dimensions"]}',
                                   '-frames', str(nframes),
                                   '-r', self.pars["fps"],
                                   '-c:v', 'libx264',
                                   '-crf', self.pars["crf"],
                                   '-c:a', 'libx264',
                                   str(Path(self.path)/f"{ctype[pos]}.mp4")],
                                   stdout=sp.DEVNULL, stderr=sp.STDOUT)
                pos += 1
