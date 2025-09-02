"""
EncoderConfigBuilder: Centralized FFmpeg command construction

Eliminates 500+ lines of duplication between build_encode_cmd and build_vbr_encode_cmd
by providing a unified interface for building encoder commands with different configurations.
"""

import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union


def ffprobe_field(path: Path, field: str) -> Optional[str]:
    """Extract a field from ffprobe output."""
    try:
        cmd = ["ffprobe", "-v", "quiet", "-select_streams", "v:0", 
               "-show_entries", f"stream={field}", "-of", "csv=p=0", str(path)]
        result = os.popen(" ".join(cmd)).read().strip()
        return result if result and result != "unknown" else None
    except:
        return None


def _extract_hdr_metadata(infile: Path) -> Tuple[Optional[str], Optional[str]]:
    """Extract HDR metadata from input file."""
    try:
        cmd = f'ffprobe -v quiet -select_streams v:0 -show_frames -read_intervals %+#1 "{infile}"'
        output = os.popen(cmd).read()
        
        master = None
        max_cll = None
        for line in output.split('\n'):
            if line.startswith("side_data_type=Mastering display metadata"):
                master = "found"  # Simplified for now
            elif line.startswith("max_content="):
                max_cll = line.split("=",1)[1].strip()
        return master, max_cll
    except Exception:
        return None, None


class EncoderConfigBuilder:
    """Builds FFmpeg encoder commands with consistent configuration across all modes."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset builder state for new command."""
        self.base_params = {}
        self.encoder_params = {}
        self.quality_params = {}
        self.threading_params = {}
        self.filter_params = {}
        self.output_params = {}
        return self
    
    def set_base_config(self, input_file: str, output_file: str, start_time: Optional[float] = None, 
                       duration: Optional[float] = None, hide_banner: bool = True, 
                       loglevel: str = "info", overwrite: bool = True) -> 'EncoderConfigBuilder':
        """Set basic FFmpeg parameters."""
        self.base_params = {
            'input_file': input_file,
            'output_file': output_file,
            'start_time': start_time,
            'duration': duration,
            'hide_banner': hide_banner,
            'loglevel': loglevel,
            'overwrite': overwrite
        }
        return self
    
    def set_encoder(self, encoder: str, preset: Optional[str] = None, crf: Optional[int] = None, 
                   bitrate: Optional[int] = None, profile: Optional[str] = None,
                   rate_control: Optional[str] = None, maxrate: Optional[int] = None,
                   bufsize: Optional[int] = None) -> 'EncoderConfigBuilder':
        """Set encoder and basic quality parameters."""
        self.encoder_params = {
            'encoder': encoder,
            'preset': preset,
            'crf': crf,
            'bitrate': bitrate,
            'profile': profile,
            'rate_control': rate_control,
            'maxrate': maxrate,
            'bufsize': bufsize
        }
        return self
    
    def set_quality_params(self, bf: Optional[int] = None, refs: Optional[int] = None, 
                          min_keyint: Optional[int] = None, keyint: Optional[int] = None,
                          rc_lookahead: Optional[int] = None, spatial_aq: Optional[bool] = None,
                          temporal_aq: Optional[bool] = None, aq_strength: Optional[float] = None,
                          preanalysis: Optional[bool] = None, vbaq: Optional[bool] = None) -> 'EncoderConfigBuilder':
        """Set quality-related encoding parameters."""
        self.quality_params = {
            'bf': bf,
            'refs': refs,
            'min_keyint': min_keyint,
            'keyint': keyint,
            'rc_lookahead': rc_lookahead,
            'spatial_aq': spatial_aq,
            'temporal_aq': temporal_aq,
            'aq_strength': aq_strength,
            'preanalysis': preanalysis,
            'vbaq': vbaq
        }
        return self
    
    def set_threading(self, frame_threads: Optional[int] = None, tile_columns: Optional[int] = None,
                     tile_rows: Optional[int] = None, wpp: Optional[bool] = None, 
                     slices: Optional[int] = None) -> 'EncoderConfigBuilder':
        """Set threading and parallelization parameters."""
        self.threading_params = {
            'frame_threads': frame_threads,
            'tile_columns': tile_columns,
            'tile_rows': tile_rows,
            'wpp': wpp,
            'slices': slices
        }
        return self
    
    def set_filters(self, scale: Optional[str] = None, crop: Optional[str] = None, 
                   deinterlace: Optional[bool] = None, custom_filters: Optional[List[str]] = None) -> 'EncoderConfigBuilder':
        """Set video filters."""
        self.filter_params = {
            'scale': scale,
            'crop': crop,
            'deinterlace': deinterlace,
            'custom_filters': custom_filters or []
        }
        return self
    
    def set_output_format(self, format: Optional[str] = None, movflags: Optional[str] = None,
                         map_streams: Optional[List[str]] = None, metadata: Optional[Dict[str, str]] = None,
                         copy_streams: bool = True, progress_pipe: bool = True,
                         preserve_hdr: bool = True, input_file: Optional[Union[str, Path]] = None) -> 'EncoderConfigBuilder':
        """Set output format and container options."""
        self.output_params = {
            'format': format,
            'movflags': movflags,
            'map_streams': map_streams or [],
            'metadata': metadata or {},
            'copy_streams': copy_streams,
            'progress_pipe': progress_pipe,
            'preserve_hdr': preserve_hdr,
            'input_file': input_file
        }
        return self
    
    def build_command(self) -> List[str]:
        """Build the complete FFmpeg command."""
        cmd = ['ffmpeg']
        
        # Base parameters
        if self.base_params.get('hide_banner'):
            cmd.append('-hide_banner')
        
        if self.base_params.get('loglevel'):
            cmd.extend(['-loglevel', self.base_params['loglevel']])
        
        if self.base_params.get('overwrite'):
            cmd.append('-y')
        
        # Input file with timing
        if self.base_params.get('start_time') is not None:
            cmd.extend(['-ss', str(self.base_params['start_time'])])
        
        cmd.extend(['-i', self.base_params['input_file']])
        
        if self.base_params.get('duration') is not None:
            cmd.extend(['-t', str(self.base_params['duration'])])
        
        # Stream mapping
        map_streams = self.output_params.get('map_streams', [])
        if not map_streams and self.output_params.get('copy_streams'):
            # Default comprehensive mapping
            cmd.extend(['-map', '0', '-map_metadata', '0', '-map_chapters', '0'])
        else:
            for stream_map in map_streams:
                cmd.extend(['-map', stream_map])
        
        # Video filters
        filters = self._build_filter_chain()
        if filters:
            cmd.extend(['-vf', filters])
        
        # Video encoder
        encoder = self.encoder_params.get('encoder')
        if encoder:
            cmd.extend(['-c:v', encoder])
        
        # Encoder-specific parameters
        cmd.extend(self._build_encoder_params())
        
        # Quality parameters
        cmd.extend(self._build_quality_params())
        
        # Threading parameters
        cmd.extend(self._build_threading_params())
        
        # HDR metadata (for hardware encoders)
        cmd.extend(self._build_hdr_metadata())
        
        # Color space parameters
        cmd.extend(self._build_color_params())
        
        # Stream copying
        if self.output_params.get('copy_streams'):
            cmd.extend(['-c:a', 'copy', '-c:s', 'copy', '-c:d', 'copy', '-c:t', 'copy', '-copy_unknown'])
        
        # Progress reporting
        if self.output_params.get('progress_pipe'):
            cmd.extend(['-progress', 'pipe:1', '-nostats'])
        
        # Output format
        if self.output_params.get('format'):
            cmd.extend(['-f', self.output_params['format']])
        
        if self.output_params.get('movflags'):
            cmd.extend(['-movflags', self.output_params['movflags']])
        
        # Metadata
        for key, value in self.output_params.get('metadata', {}).items():
            cmd.extend(['-metadata', f'{key}={value}'])
        
        # Output file
        cmd.append(self.base_params['output_file'])
        
        return cmd
    
    def _build_filter_chain(self) -> str:
        """Build video filter chain."""
        filters = []
        
        if self.filter_params.get('scale'):
            filters.append(f"scale={self.filter_params['scale']}")
        
        if self.filter_params.get('crop'):
            filters.append(f"crop={self.filter_params['crop']}")
        
        if self.filter_params.get('deinterlace'):
            filters.append('yadif')
        
        filters.extend(self.filter_params.get('custom_filters', []))
        
        return ','.join(filters) if filters else ''
    
    def _build_encoder_params(self) -> List[str]:
        """Build encoder-specific parameters."""
        params = []
        encoder = self.encoder_params.get('encoder', '')
        
        # Preset (skip for AMF encoder as it doesn't support presets)
        if self.encoder_params.get('preset') and 'amf' not in encoder:
            params.extend(['-preset', self.encoder_params['preset']])
        
        # Profile
        if self.encoder_params.get('profile'):
            params.extend(['-profile:v', self.encoder_params['profile']])
        
        # Rate control mode
        if self.encoder_params.get('rate_control'):
            params.extend(['-rc', self.encoder_params['rate_control']])
        
        # AMF specific quality setting
        if self.encoder_params.get('quality'):
            params.extend(['-quality', self.encoder_params['quality']])
        
        # Rate control parameters
        if self.encoder_params.get('crf') is not None:
            if 'nvenc' in encoder:
                params.extend(['-qp', str(self.encoder_params['crf'])])
            elif 'qsv' in encoder:
                params.extend(['-global_quality', str(self.encoder_params['crf'])])
            elif 'amf' in encoder:
                # Use stored QP values
                if self.encoder_params.get('qp_i') is not None:
                    params.extend(['-qp_i', str(self.encoder_params['qp_i']),
                                 '-qp_p', str(self.encoder_params['qp_p']),
                                 '-qp_b', str(self.encoder_params['qp_b'])])
                else:
                    params.extend(['-qp_i', str(self.encoder_params['crf']),
                                 '-qp_p', str(self.encoder_params['crf']),
                                 '-qp_b', str(self.encoder_params['crf'])])
            else:  # libx265
                params.extend(['-crf', str(self.encoder_params['crf'])])
        
        if self.encoder_params.get('bitrate') is not None:
            params.extend(['-b:v', f"{self.encoder_params['bitrate']}k"])
            
            # Add maxrate and bufsize for VBR
            if self.encoder_params.get('maxrate'):
                params.extend(['-maxrate', f"{self.encoder_params['maxrate']}k"])
            if self.encoder_params.get('bufsize'):
                params.extend(['-bufsize', f"{self.encoder_params['bufsize']}k"])
        
        # Encoder-specific additional settings
        if 'nvenc' in encoder and self.quality_params.get('bf', 0) > 0:
            params.extend(['-b_ref_mode', 'middle'])
        elif 'qsv' in encoder and self.quality_params.get('rc_lookahead'):
            params.extend(['-look_ahead', '1'])
        
        # Pixel format (auto-detected from input)
        input_file = self.output_params.get('input_file')
        if input_file:
            pixfmt = ffprobe_field(Path(input_file), "pix_fmt")
            is10 = bool(pixfmt and ("10le" in pixfmt or "p10" in pixfmt))
            pix_out = "p010le" if is10 else "nv12"
            params.extend(['-pix_fmt', pix_out])
        
        return params
    
    def _build_quality_params(self) -> List[str]:
        """Build quality-related parameters."""
        params = []
        encoder = self.encoder_params.get('encoder', '')
        
        # Keyframe intervals
        if self.quality_params.get('keyint') is not None:
            params.extend(['-g', str(self.quality_params['keyint'])])
        
        # B-frames and reference frames for hardware encoders
        if 'libx265' not in encoder:
            if self.quality_params.get('bf') is not None:
                params.extend(['-bf', str(self.quality_params['bf'])])
            if self.quality_params.get('refs') is not None:
                params.extend(['-refs', str(self.quality_params['refs'])])
        
        # Hardware encoder specific quality settings
        if 'nvenc' in encoder:
            if self.quality_params.get('rc_lookahead'):
                params.extend(['-rc-lookahead', str(self.quality_params['rc_lookahead'])])
            if self.quality_params.get('spatial_aq'):
                params.extend(['-spatial_aq', '1' if self.quality_params['spatial_aq'] else '0'])
            if self.quality_params.get('temporal_aq'):
                params.extend(['-temporal_aq', '1' if self.quality_params['temporal_aq'] else '0'])
            # NVENC AQ strength is different parameter name
            if self.quality_params.get('aq_strength'):
                params.extend(['-aq-strength', str(int(self.quality_params['aq_strength'] * 8))])
        
        elif 'amf' in encoder:
            # AMF advanced features (equivalent to use_advanced_features=True)
            if self.quality_params.get('preanalysis'):
                params.extend(['-preanalysis', '1'])
            if self.quality_params.get('vbaq'):
                params.extend(['-vbaq', '1'])
        
        # libx265 parameters via x265-params
        elif 'libx265' in encoder:
            x265_params = []
            
            # VBV constraints for VBR mode
            if self.encoder_params.get('bitrate'):
                bitrate = self.encoder_params['bitrate']
                x265_params.append(f"vbv-maxrate={bitrate}")
                x265_params.append(f"vbv-bufsize={int(bitrate * 2.0)}")
            
            # Quality parameters
            if self.quality_params.get('bf') is not None:
                x265_params.append(f"bframes={self.quality_params['bf']}")
            if self.quality_params.get('refs') is not None:
                x265_params.append(f"ref={self.quality_params['refs']}")
            if self.quality_params.get('min_keyint') is not None:
                x265_params.append(f"min-keyint={self.quality_params['min_keyint']}")
            if self.quality_params.get('rc_lookahead') is not None:
                x265_params.append(f"rc-lookahead={self.quality_params['rc_lookahead']}")
            if self.quality_params.get('aq_strength') is not None:
                x265_params.append(f"aq-strength={self.quality_params['aq_strength']}")
            
            # HDR parameters for libx265
            input_file = self.output_params.get('input_file')
            if input_file and self.output_params.get('preserve_hdr'):
                pixfmt = ffprobe_field(Path(input_file), "pix_fmt")
                is10 = bool(pixfmt and ("10le" in pixfmt or "p10" in pixfmt))
                
                if is10:
                    master_display, max_cll = _extract_hdr_metadata(Path(input_file))
                    if master_display or max_cll:
                        x265_params.extend(["hdr10=1", "hdr10-opt=1"])
                        if master_display:
                            x265_params.append(f"master-display={master_display}")
                        if max_cll:
                            x265_params.append(f"max-cll={max_cll}")
            
            # CPU optimization parameters
            x265_params.extend([
                "pools=+",  # Auto-detect optimal pool count
                "frame-threads=16",  # Increased for better parallelism
                "lookahead-threads=8",  # Increased for better CPU utilization
                "wpp=1",  # Enable Wavefront Parallel Processing
                "pmode=1",  # Enable parallel mode decision
                "pme=1",  # Enable parallel motion estimation
                "b-adapt=2",  # Adaptive B-frame placement
                "subme=7",   # High subpixel motion estimation
                "me=3",      # Uneven multi-hexagon search
            ])
            
            if x265_params:
                params.extend(['-x265-params', ':'.join(x265_params)])
        
        return params
    
    def _build_threading_params(self) -> List[str]:
        """Build threading and parallelization parameters."""
        params = []
        encoder = self.encoder_params.get('encoder', '')
        
        # Frame-level threading
        if self.threading_params.get('frame_threads') is not None:
            if 'libx265' in encoder:
                existing_x265 = self._find_x265_params(params)
                if existing_x265:
                    params[existing_x265[1]] += f":frame-threads={self.threading_params['frame_threads']}"
                else:
                    params.extend(['-x265-params', f"frame-threads={self.threading_params['frame_threads']}"])
            else:  # Hardware encoders
                params.extend(['-threads', str(self.threading_params['frame_threads'])])
        
        # Tile-based parallelization
        if self.threading_params.get('tile_columns') is not None:
            if 'nvenc' in encoder:
                params.extend(['-tile-columns', str(self.threading_params['tile_columns'])])
            elif 'libx265' in encoder:
                # x265 doesn't use tile columns directly
                pass
        
        if self.threading_params.get('tile_rows') is not None:
            if 'nvenc' in encoder:
                params.extend(['-tile-rows', str(self.threading_params['tile_rows'])])
        
        # Wavefront Parallel Processing
        if self.threading_params.get('wpp') is not None and 'libx265' in encoder:
            existing_x265 = self._find_x265_params(params)
            wpp_val = '1' if self.threading_params['wpp'] else '0'
            if existing_x265:
                params[existing_x265[1]] += f":wpp={wpp_val}"
            else:
                params.extend(['-x265-params', f"wpp={wpp_val}"])
        
        # Slices
        if self.threading_params.get('slices') is not None:
            params.extend(['-slices', str(self.threading_params['slices'])])
        
        return params
    
    def _build_hdr_metadata(self) -> List[str]:
        """Build HDR metadata parameters for hardware encoders."""
        params = []
        
        if not self.output_params.get('preserve_hdr'):
            return params
            
        input_file = self.output_params.get('input_file')
        encoder = self.encoder_params.get('encoder', '')
        
        if input_file and 'libx265' not in encoder:  # Hardware encoders only
            pixfmt = ffprobe_field(Path(input_file), "pix_fmt")
            is10 = bool(pixfmt and ("10le" in pixfmt or "p10" in pixfmt))
            
            if is10:
                master_display, max_cll = _extract_hdr_metadata(Path(input_file))
                if master_display:
                    params.extend(['-master_display', master_display])
                if max_cll:
                    params.extend(['-max_cll', max_cll])
        
        return params
    
    def _build_color_params(self) -> List[str]:
        """Build color space parameters."""
        params = []
        
        input_file = self.output_params.get('input_file')
        if input_file:
            input_path = Path(input_file)
            prim = ffprobe_field(input_path, "color_primaries")
            trc = ffprobe_field(input_path, "color_transfer")
            matrix = ffprobe_field(input_path, "colorspace")
            rng = ffprobe_field(input_path, "color_range")
            
            if prim:   params.extend(["-color_primaries:v:0", prim])
            if trc:    params.extend(["-color_trc:v:0", trc])
            if matrix: params.extend(["-colorspace:v:0", matrix])
            if rng:    params.extend(["-color_range:v:0", rng])
        
        return params
    
    def _find_x265_params(self, params: List[str]) -> Optional[Tuple[int, int]]:
        """Find existing -x265-params in command for appending."""
        try:
            idx = params.index('-x265-params')
            return (idx, idx + 1)
        except ValueError:
            return None
    
    def build_standard_encode_cmd(self, input_file: str, output_file: str, 
                                 encoder: str, preset: str, crf: int,
                                 width: int, height: int, threads: Optional[int] = None,
                                 start_time: Optional[float] = None, duration: Optional[float] = None,
                                 preserve_hdr: bool = True, debug: bool = False) -> List[str]:
        """Build standard encoding command (replaces build_encode_cmd)."""
        self.reset()
        
        # Detect pixel format and profile
        input_path = Path(input_file)
        pixfmt = ffprobe_field(input_path, "pix_fmt")
        is10 = bool(pixfmt and ("10le" in pixfmt or "p10" in pixfmt))
        profile = "main10" if is10 else "main"
        
        # Set base configuration
        loglevel = "info" if debug else "error"
        self.set_base_config(input_file, output_file, start_time, duration, 
                           loglevel=loglevel)
        
        # Set encoder with QP mapping for hardware encoders
        if 'nvenc' in encoder:
            self.set_encoder(encoder, preset, crf, rate_control='constqp', profile=profile)
        elif 'qsv' in encoder:
            self.set_encoder(encoder, preset, crf, profile=profile)
        elif 'amf' in encoder:
            self.set_encoder(encoder, rate_control='cqp', profile=profile)
            # AMF uses separate QP values
            self.encoder_params.update({
                'qp_i': crf, 'qp_p': crf, 'qp_b': crf,
                'quality': 'quality'
            })
        else:  # libx265
            self.set_encoder(encoder, preset, crf, profile=profile)
        
        # Set threading based on encoder type
        if encoder == 'libx265':
            # Software encoder threading
            frame_threads = min(threads or 16, os.cpu_count() or 16)
            self.set_threading(frame_threads=frame_threads, wpp=True)
        else:
            # Hardware encoder threading
            self.set_threading(frame_threads=min(threads or 4, 4))
        
        # Set quality parameters based on encoder
        if encoder == 'hevc_nvenc':
            self.set_quality_params(bf=3, refs=3, rc_lookahead=32, spatial_aq=True, temporal_aq=True)
        elif encoder == 'hevc_qsv':
            self.set_quality_params(bf=3, refs=3, rc_lookahead=40)
        elif encoder == 'hevc_amf':
            self.set_quality_params(bf=3, refs=3)
        elif encoder == 'libx265':
            self.set_quality_params(bf=3, refs=3, min_keyint=25, keyint=250, 
                                  rc_lookahead=25, aq_strength=1.0)
        
        # Set output format with HDR preservation
        self.set_output_format(copy_streams=True, progress_pipe=True, 
                             preserve_hdr=preserve_hdr, input_file=input_file)
        
        return self.build_command()
    
    def build_vbr_encode_cmd(self, input_file: str, output_file: str,
                            encoder: str, preset: str, bitrate: int,
                            bf: int, refs: int, width: int, height: int,
                            threads: Optional[int] = None, start_time: Optional[float] = None, 
                            duration: Optional[float] = None, preserve_hdr: bool = True,
                            debug: bool = False) -> List[str]:
        """Build VBR encoding command (replaces build_vbr_encode_cmd)."""
        self.reset()
        
        # Detect pixel format and profile
        input_path = Path(input_file)
        pixfmt = ffprobe_field(input_path, "pix_fmt")
        is10 = bool(pixfmt and ("10le" in pixfmt or "p10" in pixfmt))
        profile = "main10" if is10 else "main"
        
        # Set base configuration
        loglevel = "info" if debug else "error"
        self.set_base_config(input_file, output_file, start_time, duration, 
                           loglevel=loglevel)
        
        # Calculate VBR parameters
        if encoder == 'hevc_amf':
            maxrate = int(bitrate * 1.75)
            bufsize = int(bitrate * 2.5)
            self.set_encoder(encoder, preset, bitrate=bitrate, profile=profile,
                           rate_control='vbr_peak', maxrate=maxrate, bufsize=bufsize)
            # Add critical AMF quality parameter
            self.encoder_params['quality'] = 'quality'
        elif encoder == 'hevc_nvenc':
            maxrate = int(bitrate * 1.5)
            bufsize = int(bitrate * 2.0)
            # Map preset to NVENC preset
            nvenc_preset = 'p7' if preset == 'medium' else preset
            self.set_encoder(encoder, nvenc_preset, bitrate=bitrate, profile=profile,
                           rate_control='vbr_hq', maxrate=maxrate, bufsize=bufsize)
        elif encoder == 'hevc_qsv':
            maxrate = int(bitrate * 1.5)
            bufsize = int(bitrate * 2.0)
            self.set_encoder(encoder, preset, bitrate=bitrate, profile=profile,
                           maxrate=maxrate, bufsize=bufsize)
        else:  # libx265 - VBV constrained
            self.set_encoder(encoder, preset, bitrate=bitrate, profile=profile)
        
        # Set specific quality parameters for VBR
        self.set_quality_params(bf=bf, refs=refs, keyint=240)
        
        # Set threading based on encoder type
        if encoder == 'libx265':
            # Software encoder threading
            frame_threads = min(threads or 16, os.cpu_count() or 16)
            self.set_threading(frame_threads=frame_threads, wpp=True)
            # Additional x265 VBR optimizations
            self.quality_params.update({
                'min_keyint': 25,
                'rc_lookahead': 25,
                'aq_strength': 1.0
            })
        else:
            # Hardware encoder threading
            self.set_threading(frame_threads=min(threads or 4, 4))
            # Hardware encoder VBR optimizations
            if encoder == 'hevc_nvenc':
                self.set_quality_params(bf=bf, refs=refs, keyint=240, rc_lookahead=32, 
                                      spatial_aq=True, temporal_aq=True)
            elif encoder == 'hevc_qsv':
                self.set_quality_params(bf=bf, refs=refs, keyint=240, rc_lookahead=40)
            elif encoder == 'hevc_amf':
                # Set AMF advanced features (preanalysis and vbaq for VBR quality)
                self.set_quality_params(bf=bf, refs=refs, keyint=240, 
                                      preanalysis=True, vbaq=True)
        
        # Set output format with HDR preservation
        self.set_output_format(copy_streams=True, progress_pipe=True, 
                             preserve_hdr=preserve_hdr, input_file=input_file)
        
        return self.build_command()
    
    def build_vmaf_reference_cmd(self, input_file: str, output_file: str,
                               width: int, height: int, start_time: Optional[float] = None,
                               duration: Optional[float] = None) -> List[str]:
        """Build VMAF reference clip command."""
        self.reset()
        
        # Set base configuration
        self.set_base_config(input_file, output_file, start_time, duration)
        
        # Set encoder for reference (high quality)
        self.set_encoder('libx265', 'medium', crf=18)
        
        # Set high-quality parameters
        self.set_quality_params(bf=4, refs=4, min_keyint=25, keyint=250, 
                              rc_lookahead=32, aq_strength=1.0)
        
        # Set threading
        frame_threads = min(16, os.cpu_count() or 16)
        self.set_threading(frame_threads=frame_threads, wpp=True)
        
        # Set output format
        self.set_output_format(map_streams=['0:v'])
        
        return self.build_command()
    
    def build_vmaf_comparison_cmd(self, encoded_file: str, reference_file: str,
                                 threads: Optional[int] = None) -> List[str]:
        """Build VMAF comparison command."""
        vmaf_threads = min(threads or 16, os.cpu_count() or 16)
        
        return [
            'ffmpeg', '-hide_banner', '-loglevel', 'info', '-y',
            '-i', encoded_file,
            '-i', reference_file,
            '-lavfi', f'[0:v][1:v]libvmaf=n_threads={vmaf_threads}:n_subsample=1',
            '-f', 'null', '-'
        ]
    
    def get_encoder_profile_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get predefined encoder configurations."""
        return {
            'hevc_nvenc': {
                'fast': {'preset': 'fast', 'bf': 2, 'refs': 2},
                'medium': {'preset': 'medium', 'bf': 3, 'refs': 3},
                'slow': {'preset': 'slow', 'bf': 4, 'refs': 4}
            },
            'hevc_qsv': {
                'fast': {'preset': 'fast', 'bf': 2, 'refs': 2},
                'medium': {'preset': 'medium', 'bf': 3, 'refs': 3},
                'slow': {'preset': 'slow', 'bf': 4, 'refs': 4}
            },
            'hevc_amf': {
                'fast': {'preset': 'speed', 'bf': 2, 'refs': 2},
                'medium': {'preset': 'balanced', 'bf': 3, 'refs': 3},
                'slow': {'preset': 'quality', 'bf': 4, 'refs': 4}
            },
            'libx265': {
                'fast': {'preset': 'fast', 'bf': 2, 'refs': 2},
                'medium': {'preset': 'medium', 'bf': 3, 'refs': 3},
                'slow': {'preset': 'slow', 'bf': 4, 'refs': 4}
            }
        }
