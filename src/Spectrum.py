"""This code is a modification of https://github.com/AndMastro/WreckingNet and was provided to us by the authors"""



import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import librosa
import librosa.display
import sys
import os

plt.ioff()

class Spectrum:

    @staticmethod
    def plot_spectrogram(rate, data, NFFT=1024, noverlap=256):
        """
        :param rate: int
            Sample rate of the audio file
        :param data: numpy array
            Data to plot
        :param NFFT: int
            The number of data points used in each block for the FFT.
        :param noverlap: int
         The number of points of overlap between blocks
        :return: Figure
        """
        fig, ax = plt.subplots(1)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.axis('off')
        ax.specgram(data, NFFT=NFFT, noverlap=noverlap, Fs=rate)
        ax.axis('off')
        return fig

    @staticmethod
    def get_specgram_mplib(path, file_name, out_path, fmt="png", NFFT=1024, noverlap=512):
        """
        :param path: str
            path to the .wav file to generate spectrogram
        :param fmt: str
            name of the format to save the plot, default svg
            the format available are the same supported by matplotlib.pyplot.savefig
        :param NFFT: int
            The number of data points used in each block for the FFT.
        :param noverlap: int
            The number of points of overlap between blocks
        :return: None
            creates a image into the path with the same name of input file
            If the data has more channels, creates more svg files.
        """

        out_name = path
        if path.endswith(".wav"):
            out_name = out_path+file_name
        rate, data = wavfile.read(path)
        if data.ndim > 1:
            # we have more than one channel, we have to do more plots
            for i in range(0, data.ndim):
                dimension_data = data[:, i]
                out_path = out_name + "_channel_" + str(i) + "." + fmt
                fig = Spectrum.plot_spectrogram(rate, dimension_data, NFFT, noverlap);
                fig.savefig(out_path, format=fmt, frameon='false')
                plt.close(fig)
        else:
            out_name += "." + fmt
            fig = Spectrum.plot_spectrogram(rate, data, NFFT, noverlap);
            fig.savefig(out_name, format=fmt, frameon='false')
            plt.close(fig)

    @staticmethod
    def compute_specgram_and_delta(path, sample_rate=44100, normalize=True, nfft=1024, hop_len=512, n_mel_bands=60):
        """
        :param path: str
            path where the wav file is located
        :param normalize: bool
            Normalize the data before computing the spectogram
        :param sample_rate: int
            sample rate to re-sample the wav.
            default is set to audible frequencies
        :param nfft: int
            The number of data points used in each block for the FFT.
        :param hop_len: int
            The number of points of overlap between blocks
        :param n_mel_bands: int
            number of Mel bands to generate
        :return: np.array
            log-scale mel spectrogram stacked with its delta
        """

        signal, fs = librosa.load(path, sr=sample_rate)
        if normalize:
            signal = librosa.util.normalize(signal)

        spec = librosa.feature.melspectrogram(y=signal, sr=fs, n_fft=nfft, hop_length=hop_len, n_mels=n_mel_bands)

        # generating first channel, log-scaled mel spectrogram (default parameters match the ones used in the paper)
        log_spec = librosa.power_to_db(spec, ref=np.max)
        # delta computation, second channel
        delta_log_spec = librosa.feature.delta(log_spec, mode='nearest')

        return np.stack([log_spec, delta_log_spec], axis=-1)

    @staticmethod
    def get_specgram_librosa(path, file_name, out_path, fmt='png', which='both', sample_rate=44100, nfft=1024, hop_len=512, n_mel_bands=60):
        """
        :param path: str
            path where the wav file is located
        :param fmt: str
            format to save the image
        :param which: str {log, delta, both}
            which spectrogram to save
        :param sample_rate:
            sample rate to re-sample the wav.
            default is set to audible frequencies
        :param nfft: int
            The number of data points used in each block for the FFT.
        :param hop_len: int
            The number of points of overlap between blocks
        :param n_mel_bands:
            number of Mel bands to generate
        :return: None
            creates two images into the path with the same name of input file
        """
        out_name = path
        if path.endswith(".wav"):
            out_name = out_path+file_name

        spec = Spectrum.compute_specgram_and_delta(path, sample_rate, nfft, hop_len, n_mel_bands)
        log_spec = spec[:, :, 0]
        #delta_spec = spec[:, :, 1]

        if which != 'delta':
            librosa.display.specshow(log_spec)
            out_path = out_name + '.' + fmt
            plt.savefig(out_path, format=fmt, frameon='false', bbox_inches='tight', pad_inches=0)

        #if which != 'log':
            #librosa.display.specshow(delta_spec)
            #out_path = out_name + '_delta.' + fmt
            #plt.savefig(out_path, format=fmt, frameon='false', bbox_inches='tight', pad_inches=0)


# fft - fast fourier transform
def main():
    # generate spectograms from segments

    #folders = ["testing0","testing1","testing2","testing3","testing4","training0","training1","training2","training3","training4"]


    all_segments = ["training0"]
    all_segments.sort()
    tot_segments = len(all_segments)
    n_directory = 0
    print(all_segments)
    for directory in all_segments:
        print(str(n_directory)+"/"+str(tot_segments)+" - "+directory)

        audio_path = '../segments/' + directory
        img_path = '../segments-spectogram/' + directory

        all_category = os.listdir(audio_path)
        all_category.sort()
        print(all_category)

        n_directory += 1
        tot_category = len(all_category)
        n_category = 0

        for category in all_category:
            print(str(n_category) + "/" + str(tot_category) + " - " + category)

            all_sub_category = os.listdir(audio_path + '/' + category)
            all_sub_category.sort()
            print(all_sub_category)

            n_category += 1
            tot_sub_category = len(all_sub_category)
            n_sub_category = 0


            for sub_category in all_sub_category:
                print(str(n_sub_category) + "/" + str(tot_sub_category) + " - " + sub_category)
                print("\n"+directory + '/' + category + '/' + sub_category)
                all_tracks = os.listdir(audio_path + '/' + category + '/' + sub_category + '/')

                n_sub_category += 1
                tot_tracks = len(all_tracks)
                n_track = 0

                for track in all_tracks:
                    sys.stdout.write('\r'+str(n_track)+"/"+str(tot_tracks)+" - "+track)
                    sys.stdout.flush()

                    wav_path = audio_path + '/' + category + '/' + sub_category + '/' + track
                    spectogram_path = img_path + '/' + category + '/' + sub_category + '/'
                    #generation of spectrogram for all channels in wav
                    #Spectrum.get_specgram_mplib(wav_path, track[:-4], spectogram_path)

                    #generation of spectrum based on delta
                    Spectrum.get_specgram_mplib(wav_path, track[:-4], spectogram_path)
                    n_track += 1


if __name__ == "__main__":
    main()
