<div class="cell markdown" data-colab_type="text" id="view-in-github">

<a href="https://colab.research.google.com/github/raj911tx/covid-19_detection/blob/master/Covid19.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

</div>

<div class="cell markdown" data-colab_type="text" id="wht8wNuSNrNX">

# CoronaHack -Chest X-Ray-Dataset

-----

## *Classify the X Ray image which is having Corona*

</div>

<div class="cell code" data-execution_count="0" data-colab="{}" data-colab_type="code" id="0rzPEftR9cPL">

``` python
from google.colab import files
!pip install -q kaggle
```

</div>

<div class="cell markdown" data-colab_type="text" id="D3AsW_MGDAmt">

# Run the following command to access the Kaggle API using the command line: pip install kaggle

</div>

<div class="cell code" data-execution_count="0" data-colab="{&quot;height&quot;:75,&quot;resources&quot;:{&quot;http://localhost:8080/nbextensions/google.colab/files.js&quot;:{&quot;ok&quot;:true,&quot;status&quot;:200,&quot;data&quot;:&quot;Ly8gQ29weXJpZ2h0IDIwMTcgR29vZ2xlIExMQwovLwovLyBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgIkxpY2Vuc2UiKTsKLy8geW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLgovLyBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXQKLy8KLy8gICAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjAKLy8KLy8gVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZQovLyBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiAiQVMgSVMiIEJBU0lTLAovLyBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC4KLy8gU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZAovLyBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS4KCi8qKgogKiBAZmlsZW92ZXJ2aWV3IEhlbHBlcnMgZm9yIGdvb2dsZS5jb2xhYiBQeXRob24gbW9kdWxlLgogKi8KKGZ1bmN0aW9uKHNjb3BlKSB7CmZ1bmN0aW9uIHNwYW4odGV4dCwgc3R5bGVBdHRyaWJ1dGVzID0ge30pIHsKICBjb25zdCBlbGVtZW50ID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudCgnc3BhbicpOwogIGVsZW1lbnQudGV4dENvbnRlbnQgPSB0ZXh0OwogIGZvciAoY29uc3Qga2V5IG9mIE9iamVjdC5rZXlzKHN0eWxlQXR0cmlidXRlcykpIHsKICAgIGVsZW1lbnQuc3R5bGVba2V5XSA9IHN0eWxlQXR0cmlidXRlc1trZXldOwogIH0KICByZXR1cm4gZWxlbWVudDsKfQoKLy8gTWF4IG51bWJlciBvZiBieXRlcyB3aGljaCB3aWxsIGJlIHVwbG9hZGVkIGF0IGEgdGltZS4KY29uc3QgTUFYX1BBWUxPQURfU0laRSA9IDEwMCAqIDEwMjQ7Ci8vIE1heCBhbW91bnQgb2YgdGltZSB0byBibG9jayB3YWl0aW5nIGZvciB0aGUgdXNlci4KY29uc3QgRklMRV9DSEFOR0VfVElNRU9VVF9NUyA9IDMwICogMTAwMDsKCmZ1bmN0aW9uIF91cGxvYWRGaWxlcyhpbnB1dElkLCBvdXRwdXRJZCkgewogIGNvbnN0IHN0ZXBzID0gdXBsb2FkRmlsZXNTdGVwKGlucHV0SWQsIG91dHB1dElkKTsKICBjb25zdCBvdXRwdXRFbGVtZW50ID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQob3V0cHV0SWQpOwogIC8vIENhY2hlIHN0ZXBzIG9uIHRoZSBvdXRwdXRFbGVtZW50IHRvIG1ha2UgaXQgYXZhaWxhYmxlIGZvciB0aGUgbmV4dCBjYWxsCiAgLy8gdG8gdXBsb2FkRmlsZXNDb250aW51ZSBmcm9tIFB5dGhvbi4KICBvdXRwdXRFbGVtZW50LnN0ZXBzID0gc3RlcHM7CgogIHJldHVybiBfdXBsb2FkRmlsZXNDb250aW51ZShvdXRwdXRJZCk7Cn0KCi8vIFRoaXMgaXMgcm91Z2hseSBhbiBhc3luYyBnZW5lcmF0b3IgKG5vdCBzdXBwb3J0ZWQgaW4gdGhlIGJyb3dzZXIgeWV0KSwKLy8gd2hlcmUgdGhlcmUgYXJlIG11bHRpcGxlIGFzeW5jaHJvbm91cyBzdGVwcyBhbmQgdGhlIFB5dGhvbiBzaWRlIGlzIGdvaW5nCi8vIHRvIHBvbGwgZm9yIGNvbXBsZXRpb24gb2YgZWFjaCBzdGVwLgovLyBUaGlzIHVzZXMgYSBQcm9taXNlIHRvIGJsb2NrIHRoZSBweXRob24gc2lkZSBvbiBjb21wbGV0aW9uIG9mIGVhY2ggc3RlcCwKLy8gdGhlbiBwYXNzZXMgdGhlIHJlc3VsdCBvZiB0aGUgcHJldmlvdXMgc3RlcCBhcyB0aGUgaW5wdXQgdG8gdGhlIG5leHQgc3RlcC4KZnVuY3Rpb24gX3VwbG9hZEZpbGVzQ29udGludWUob3V0cHV0SWQpIHsKICBjb25zdCBvdXRwdXRFbGVtZW50ID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQob3V0cHV0SWQpOwogIGNvbnN0IHN0ZXBzID0gb3V0cHV0RWxlbWVudC5zdGVwczsKCiAgY29uc3QgbmV4dCA9IHN0ZXBzLm5leHQob3V0cHV0RWxlbWVudC5sYXN0UHJvbWlzZVZhbHVlKTsKICByZXR1cm4gUHJvbWlzZS5yZXNvbHZlKG5leHQudmFsdWUucHJvbWlzZSkudGhlbigodmFsdWUpID0+IHsKICAgIC8vIENhY2hlIHRoZSBsYXN0IHByb21pc2UgdmFsdWUgdG8gbWFrZSBpdCBhdmFpbGFibGUgdG8gdGhlIG5leHQKICAgIC8vIHN0ZXAgb2YgdGhlIGdlbmVyYXRvci4KICAgIG91dHB1dEVsZW1lbnQubGFzdFByb21pc2VWYWx1ZSA9IHZhbHVlOwogICAgcmV0dXJuIG5leHQudmFsdWUucmVzcG9uc2U7CiAgfSk7Cn0KCi8qKgogKiBHZW5lcmF0b3IgZnVuY3Rpb24gd2hpY2ggaXMgY2FsbGVkIGJldHdlZW4gZWFjaCBhc3luYyBzdGVwIG9mIHRoZSB1cGxvYWQKICogcHJvY2Vzcy4KICogQHBhcmFtIHtzdHJpbmd9IGlucHV0SWQgRWxlbWVudCBJRCBvZiB0aGUgaW5wdXQgZmlsZSBwaWNrZXIgZWxlbWVudC4KICogQHBhcmFtIHtzdHJpbmd9IG91dHB1dElkIEVsZW1lbnQgSUQgb2YgdGhlIG91dHB1dCBkaXNwbGF5LgogKiBAcmV0dXJuIHshSXRlcmFibGU8IU9iamVjdD59IEl0ZXJhYmxlIG9mIG5leHQgc3RlcHMuCiAqLwpmdW5jdGlvbiogdXBsb2FkRmlsZXNTdGVwKGlucHV0SWQsIG91dHB1dElkKSB7CiAgY29uc3QgaW5wdXRFbGVtZW50ID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoaW5wdXRJZCk7CiAgaW5wdXRFbGVtZW50LmRpc2FibGVkID0gZmFsc2U7CgogIGNvbnN0IG91dHB1dEVsZW1lbnQgPSBkb2N1bWVudC5nZXRFbGVtZW50QnlJZChvdXRwdXRJZCk7CiAgb3V0cHV0RWxlbWVudC5pbm5lckhUTUwgPSAnJzsKCiAgY29uc3QgcGlja2VkUHJvbWlzZSA9IG5ldyBQcm9taXNlKChyZXNvbHZlKSA9PiB7CiAgICBpbnB1dEVsZW1lbnQuYWRkRXZlbnRMaXN0ZW5lcignY2hhbmdlJywgKGUpID0+IHsKICAgICAgcmVzb2x2ZShlLnRhcmdldC5maWxlcyk7CiAgICB9KTsKICB9KTsKCiAgY29uc3QgY2FuY2VsID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudCgnYnV0dG9uJyk7CiAgaW5wdXRFbGVtZW50LnBhcmVudEVsZW1lbnQuYXBwZW5kQ2hpbGQoY2FuY2VsKTsKICBjYW5jZWwudGV4dENvbnRlbnQgPSAnQ2FuY2VsIHVwbG9hZCc7CiAgY29uc3QgY2FuY2VsUHJvbWlzZSA9IG5ldyBQcm9taXNlKChyZXNvbHZlKSA9PiB7CiAgICBjYW5jZWwub25jbGljayA9ICgpID0+IHsKICAgICAgcmVzb2x2ZShudWxsKTsKICAgIH07CiAgfSk7CgogIC8vIENhbmNlbCB1cGxvYWQgaWYgdXNlciBoYXNuJ3QgcGlja2VkIGFueXRoaW5nIGluIHRpbWVvdXQuCiAgY29uc3QgdGltZW91dFByb21pc2UgPSBuZXcgUHJvbWlzZSgocmVzb2x2ZSkgPT4gewogICAgc2V0VGltZW91dCgoKSA9PiB7CiAgICAgIHJlc29sdmUobnVsbCk7CiAgICB9LCBGSUxFX0NIQU5HRV9USU1FT1VUX01TKTsKICB9KTsKCiAgLy8gV2FpdCBmb3IgdGhlIHVzZXIgdG8gcGljayB0aGUgZmlsZXMuCiAgY29uc3QgZmlsZXMgPSB5aWVsZCB7CiAgICBwcm9taXNlOiBQcm9taXNlLnJhY2UoW3BpY2tlZFByb21pc2UsIHRpbWVvdXRQcm9taXNlLCBjYW5jZWxQcm9taXNlXSksCiAgICByZXNwb25zZTogewogICAgICBhY3Rpb246ICdzdGFydGluZycsCiAgICB9CiAgfTsKCiAgaWYgKCFmaWxlcykgewogICAgcmV0dXJuIHsKICAgICAgcmVzcG9uc2U6IHsKICAgICAgICBhY3Rpb246ICdjb21wbGV0ZScsCiAgICAgIH0KICAgIH07CiAgfQoKICBjYW5jZWwucmVtb3ZlKCk7CgogIC8vIERpc2FibGUgdGhlIGlucHV0IGVsZW1lbnQgc2luY2UgZnVydGhlciBwaWNrcyBhcmUgbm90IGFsbG93ZWQuCiAgaW5wdXRFbGVtZW50LmRpc2FibGVkID0gdHJ1ZTsKCiAgZm9yIChjb25zdCBmaWxlIG9mIGZpbGVzKSB7CiAgICBjb25zdCBsaSA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoJ2xpJyk7CiAgICBsaS5hcHBlbmQoc3BhbihmaWxlLm5hbWUsIHtmb250V2VpZ2h0OiAnYm9sZCd9KSk7CiAgICBsaS5hcHBlbmQoc3BhbigKICAgICAgICBgKCR7ZmlsZS50eXBlIHx8ICduL2EnfSkgLSAke2ZpbGUuc2l6ZX0gYnl0ZXMsIGAgKwogICAgICAgIGBsYXN0IG1vZGlmaWVkOiAkewogICAgICAgICAgICBmaWxlLmxhc3RNb2RpZmllZERhdGUgPyBmaWxlLmxhc3RNb2RpZmllZERhdGUudG9Mb2NhbGVEYXRlU3RyaW5nKCkgOgogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAnbi9hJ30gLSBgKSk7CiAgICBjb25zdCBwZXJjZW50ID0gc3BhbignMCUgZG9uZScpOwogICAgbGkuYXBwZW5kQ2hpbGQocGVyY2VudCk7CgogICAgb3V0cHV0RWxlbWVudC5hcHBlbmRDaGlsZChsaSk7CgogICAgY29uc3QgZmlsZURhdGFQcm9taXNlID0gbmV3IFByb21pc2UoKHJlc29sdmUpID0+IHsKICAgICAgY29uc3QgcmVhZGVyID0gbmV3IEZpbGVSZWFkZXIoKTsKICAgICAgcmVhZGVyLm9ubG9hZCA9IChlKSA9PiB7CiAgICAgICAgcmVzb2x2ZShlLnRhcmdldC5yZXN1bHQpOwogICAgICB9OwogICAgICByZWFkZXIucmVhZEFzQXJyYXlCdWZmZXIoZmlsZSk7CiAgICB9KTsKICAgIC8vIFdhaXQgZm9yIHRoZSBkYXRhIHRvIGJlIHJlYWR5LgogICAgbGV0IGZpbGVEYXRhID0geWllbGQgewogICAgICBwcm9taXNlOiBmaWxlRGF0YVByb21pc2UsCiAgICAgIHJlc3BvbnNlOiB7CiAgICAgICAgYWN0aW9uOiAnY29udGludWUnLAogICAgICB9CiAgICB9OwoKICAgIC8vIFVzZSBhIGNodW5rZWQgc2VuZGluZyB0byBhdm9pZCBtZXNzYWdlIHNpemUgbGltaXRzLiBTZWUgYi82MjExNTY2MC4KICAgIGxldCBwb3NpdGlvbiA9IDA7CiAgICB3aGlsZSAocG9zaXRpb24gPCBmaWxlRGF0YS5ieXRlTGVuZ3RoKSB7CiAgICAgIGNvbnN0IGxlbmd0aCA9IE1hdGgubWluKGZpbGVEYXRhLmJ5dGVMZW5ndGggLSBwb3NpdGlvbiwgTUFYX1BBWUxPQURfU0laRSk7CiAgICAgIGNvbnN0IGNodW5rID0gbmV3IFVpbnQ4QXJyYXkoZmlsZURhdGEsIHBvc2l0aW9uLCBsZW5ndGgpOwogICAgICBwb3NpdGlvbiArPSBsZW5ndGg7CgogICAgICBjb25zdCBiYXNlNjQgPSBidG9hKFN0cmluZy5mcm9tQ2hhckNvZGUuYXBwbHkobnVsbCwgY2h1bmspKTsKICAgICAgeWllbGQgewogICAgICAgIHJlc3BvbnNlOiB7CiAgICAgICAgICBhY3Rpb246ICdhcHBlbmQnLAogICAgICAgICAgZmlsZTogZmlsZS5uYW1lLAogICAgICAgICAgZGF0YTogYmFzZTY0LAogICAgICAgIH0sCiAgICAgIH07CiAgICAgIHBlcmNlbnQudGV4dENvbnRlbnQgPQogICAgICAgICAgYCR7TWF0aC5yb3VuZCgocG9zaXRpb24gLyBmaWxlRGF0YS5ieXRlTGVuZ3RoKSAqIDEwMCl9JSBkb25lYDsKICAgIH0KICB9CgogIC8vIEFsbCBkb25lLgogIHlpZWxkIHsKICAgIHJlc3BvbnNlOiB7CiAgICAgIGFjdGlvbjogJ2NvbXBsZXRlJywKICAgIH0KICB9Owp9CgpzY29wZS5nb29nbGUgPSBzY29wZS5nb29nbGUgfHwge307CnNjb3BlLmdvb2dsZS5jb2xhYiA9IHNjb3BlLmdvb2dsZS5jb2xhYiB8fCB7fTsKc2NvcGUuZ29vZ2xlLmNvbGFiLl9maWxlcyA9IHsKICBfdXBsb2FkRmlsZXMsCiAgX3VwbG9hZEZpbGVzQ29udGludWUsCn07Cn0pKHNlbGYpOwo=&quot;,&quot;status_text&quot;:&quot;&quot;,&quot;headers&quot;:[[&quot;content-type&quot;,&quot;application/javascript&quot;]]}},&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" data-colab_type="code" id="Lgbux12n9sAA" data-outputId="5dcdba75-a0d6-44f9-bc7a-c917707edb4a">

``` python
uploaded=files.upload()
```

<div class="output display_data">

    <IPython.core.display.HTML object>

</div>

<div class="output stream stdout">

    Saving kaggle.json to kaggle.json

</div>

</div>

<div class="cell markdown" data-colab_type="text" id="aDOR1E8XDWIY">

## Now to import Api key from kaggle after downloading the .json file by running thsi command

</div>

<div class="cell markdown" data-colab_type="text" id="bUjzRWZUDQ5O">

</div>

<div class="cell code" data-execution_count="0" data-colab="{&quot;height&quot;:35,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" data-colab_type="code" id="U7agQXu1AQGG" data-outputId="1ffc08db-0052-4cc2-a4a0-e61c80380239">

``` python
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!ls ~/.kaggle
!chmod 600 /root/.kaggle/kaggle.json
```

<div class="output stream stdout">

    kaggle.json

</div>

</div>

<div class="cell markdown" data-colab_type="text" id="mey3QRvbEA6F">

# Make a root kaggle directory and move that .json file into that dir

-----

</div>

<div class="cell code" data-execution_count="0" data-colab="{&quot;height&quot;:70,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" data-colab_type="code" id="AuIbI3mB_RlY" data-outputId="cd6c441e-c53c-4837-bd40-696ef95138e0">

``` python
!kaggle datasets download -d praveengovi/coronahack-chest-xraydataset
```

<div class="output stream stdout">

    Downloading coronahack-chest-xraydataset.zip to /content
    100% 1.19G/1.19G [00:18<00:00, 85.8MB/s]
    100% 1.19G/1.19G [00:18<00:00, 68.0MB/s]

</div>

</div>

<div class="cell markdown" data-colab_type="text" id="Cu-3kth_ELGM">

Now download that dataset from
kaggle![](https://drive.google.com/uc?id=1Nx7i5p8PbM4K8UXLMfUD50trwGhOUdej)

</div>

<div class="cell code" data-execution_count="0" data-colab="{&quot;height&quot;:35,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" data-colab_type="code" id="8ZipEN-oDg5_" data-outputId="6cef9cdf-3bfa-41a0-df01-8e521bde6a09">

``` python
!ls
!unzip -q coronahack-chest-xraydataset.zip
```

<div class="output stream stdout">

    coronahack-chest-xraydataset.zip  kaggle.json  sample_data

</div>

</div>

<div class="cell markdown" data-colab_type="text" id="QbwNqsWdFvPV">

Unzip the downloaded files to be used for the Project

</div>

<div class="cell code" data-execution_count="0" data-colab="{&quot;height&quot;:70,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" data-colab_type="code" id="rwLnTo1jFRrF" data-outputId="a85f0599-88c6-4867-cf21-7a621a43f82c">

``` python
!ls
```

<div class="output stream stdout">

    Chest_xray_Corona_dataset_Summary.csv  coronahack-chest-xraydataset.zip
    Chest_xray_Corona_Metadata.csv	       kaggle.json
    Coronahack-Chest-XRay-Dataset	       sample_data

</div>

</div>

<div class="cell markdown" data-colab_type="text" id="rv4f_z_lGJvv">

![](https://drive.google.com/open?id=1XIQaqRR1bELwnnVslJjPMTNv3EGnAYkq)

</div>

<div class="cell markdown" data-colab_type="text" id="NHKj5eRIF1oG">

The zip file Contains he following .csv file and the train and test data
set
![](https://drive.google.com/uc?id=1XIQaqRR1bELwnnVslJjPMTNv3EGnAYkq)
---

</div>

<div class="cell code" data-execution_count="0" data-colab="{}" data-colab_type="code" id="siZj7bYoN_ri">

``` python
TRAIN_DIR = '/content/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train'
TEST_DIR = '/content/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test'
```

</div>

<div class="cell code" data-execution_count="0" data-colab="{&quot;height&quot;:194,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" data-colab_type="code" id="AgwHD3VyTfot" data-outputId="558394d0-8efb-4693-d205-6a9a0fe3beda">

``` python
import pandas as pd
metadata = pd.read_csv('Chest_xray_Corona_Metadata.csv')
metadata.head()
```

<div class="output execute_result" data-execution_count="11">

``` 
   Unnamed: 0   X_ray_image_name  ... Label_2_Virus_category Label_1_Virus_category
0           0  IM-0128-0001.jpeg  ...                    NaN                    NaN
1           1  IM-0127-0001.jpeg  ...                    NaN                    NaN
2           2  IM-0125-0001.jpeg  ...                    NaN                    NaN
3           3  IM-0122-0001.jpeg  ...                    NaN                    NaN
4           4  IM-0119-0001.jpeg  ...                    NaN                    NaN

[5 rows x 6 columns]
```

</div>

</div>

<div class="cell markdown" data-colab_type="text" id="pwwJEw4hHayE">

![](https://drive.google.com/open?id=1ORukZs5Z5dJ08HaHj4mCftavwygdjp72)
![](https://drive.google.com/open?id=1Iisi74QIWY6CSAYJrf3_lTiSCJOJEq4u)

</div>

<div class="cell code" data-execution_count="0" data-colab="{&quot;height&quot;:298,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" data-colab_type="code" id="oPbGP4hX0Mgw" data-outputId="a2e43329-17ec-4bd6-a606-2ccc6c5b9b90">

``` python
import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(x="Label_2_Virus_category",y="Label_1_Virus_category",hue="Label",data=metadata)
```

<div class="output execute_result" data-execution_count="29">

    <matplotlib.axes._subplots.AxesSubplot at 0x7f135a5f3390>

</div>

<div class="output display_data">

![](7adbb0df0d4ac760c66c76c3a2dcd0298e7ad191.png)

</div>

</div>

<div class="cell markdown" data-colab_type="text" id="O45-uyL7Hq0F">

Relationships are plotted between Disease name and their catagory

</div>

<div class="cell markdown" data-colab_type="text" id="nAvhl2T9H1ce">

Unique labels are outputted here

-----

![](https://drive.google.com/uc?id=1nAYxuYno6F8HTPPLA1IY_9au7VGJRGUA)

</div>

<div class="cell markdown" data-colab_type="text" id="izPcAYxgLWLX">

![](https://drive.google.com/uc?id=1e5a5ECva3EU9xSB7TAp0P-MJql7FXp1q)

</div>

<div class="cell code" data-execution_count="0" data-colab="{&quot;height&quot;:35,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" data-colab_type="code" id="MOMyCeE1hy3E" data-outputId="9d18bbbb-9a62-40f4-e2b1-65260fde014d">

``` python
metadata["Label"].unique()
```

<div class="output execute_result" data-execution_count="30">

    array(['Normal', 'Pnemonia'], dtype=object)

</div>

</div>

<div class="cell code" data-execution_count="0" data-colab="{&quot;height&quot;:35,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" data-colab_type="code" id="Fb31O4N5iERX" data-outputId="83ba090d-fc4d-4067-c1c9-29c26f0e3896">

``` python
metadata["Label_2_Virus_category"].unique()
```

<div class="output execute_result" data-execution_count="31">

    array([nan, 'Streptococcus', 'COVID-19', 'ARDS', 'SARS'], dtype=object)

</div>

</div>

<div class="cell code" data-execution_count="0" data-colab="{&quot;height&quot;:35,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" data-colab_type="code" id="Y3qyFG5piY6N" data-outputId="c50d0d91-eb5f-4789-eb7f-eecb56ad26d1">

``` python
metadata["Label_1_Virus_category"].unique()
```

<div class="output execute_result" data-execution_count="32">

    array([nan, 'Virus', 'bacteria', 'Stress-Smoking'], dtype=object)

</div>

</div>

<div class="cell markdown" data-colab_type="text" id="IWF0P9Z6H-Ei">

## .csv file is the cleaned up to extract desired Features for the deep learning step to get maximum efficiency

</div>

<div class="cell code" data-execution_count="0" data-colab="{&quot;height&quot;:399,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" data-colab_type="code" id="SmK8OaAbhdzP" data-outputId="5273cc32-a75c-45ff-984d-0f3a48b4055d">

``` python
meta_train_normal=metadata[(metadata["Label"]=="Normal")&(metadata["Dataset_type"]=="TRAIN")]
meta_train_pnemonia=metadata[(metadata["Label"]=="Pnemonia")&(metadata["Dataset_type"]=="TRAIN")&(metadata["Label_1_Virus_category"]=="Virus")&(metadata["Label_2_Virus_category"].isnull())]
meta_train_covid=metadata[(metadata["Label"]=="Pnemonia")&(metadata["Dataset_type"]=="TRAIN")&(metadata["Label_2_Virus_category"]=="COVID-19")&(metadata["Label_1_Virus_category"]=="Virus")]

meta_test_normal=metadata[(metadata["Label"]=="Normal")&(metadata["Dataset_type"]=="TEST")]
meta_test_pnemonia=metadata[(metadata["Label"]=="Pnemonia")&(metadata["Dataset_type"]=="TEST")&(metadata["Label_1_Virus_category"]=="Virus")&(metadata["Label_2_Virus_category"].isnull())]
meta_test_covid=metadata[(metadata["Label"]=="Pnemonia")&(metadata["Dataset_type"]=="TEST")&(metadata["Label_2_Virus_category"]=="COVID-19")&(metadata["Label_1_Virus_category"]=="Virus")]
#meta_train_normal.count()
#meta_train_pnemonia.count()
#meta_train_covid.count()
#meta_train_pnemonia.head()
#meta_test_covid.count()
meta_train_normal=meta_train_normal.assign(Class=meta_train_normal["Label"])
meta_train_pnemonia=meta_train_pnemonia.assign(Class=meta_train_pnemonia["Label"])
meta_train_covid=meta_train_covid.assign(Class=meta_train_covid["Label_2_Virus_category"])
meta_train=pd.concat([meta_train_normal,meta_train_pnemonia,meta_train_covid])
meta_train=meta_train[["X_ray_image_name","Class"]]
meta_train
meta_test_normal=meta_test_normal.assign(Class=meta_test_normal["Label"])
meta_test_pnemonia=meta_test_pnemonia.assign(Class=meta_test_pnemonia["Label"])
meta_test_covid=meta_test_covid.assign(Class=meta_test_covid["Label_2_Virus_category"])
meta_test=pd.concat([meta_test_normal,meta_test_pnemonia,meta_test_covid])
meta_test=meta_test[["X_ray_image_name","Class"]]
meta_test
```

<div class="output execute_result" data-execution_count="33">

``` 
                X_ray_image_name     Class
5286           IM-0021-0001.jpeg    Normal
5287           IM-0019-0001.jpeg    Normal
5288           IM-0017-0001.jpeg    Normal
5289           IM-0016-0001.jpeg    Normal
5290           IM-0015-0001.jpeg    Normal
...                          ...       ...
5905  person1637_virus_2834.jpeg  Pnemonia
5906  person1635_virus_2831.jpeg  Pnemonia
5907  person1634_virus_2830.jpeg  Pnemonia
5908  person1633_virus_2829.jpeg  Pnemonia
5909  person1632_virus_2827.jpeg  Pnemonia

[382 rows x 2 columns]
```

</div>

</div>

<div class="cell markdown" data-colab_type="text" id="WXCbnqKXIcnt">

## Now save split the train test labels in the following files using pandas to\_csv() command

</div>

<div class="cell code" data-execution_count="0" data-colab="{}" data-colab_type="code" id="rwZiHQVILbt6">

``` python
meta_train.to_csv("train.csv")
meta_test.to_csv("test.csv")
```

</div>

<div class="cell markdown" data-colab_type="text" id="39r_R9ElIquY">

### **We can see two new files are added,now delete the zip file as we dont need them now**

</div>

<div class="cell code" data-execution_count="0" data-colab="{&quot;height&quot;:88,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" data-colab_type="code" id="iOHy7pABL2GI" data-outputId="dc6a36ed-c9f1-4fc8-c428-9c5c2b4a0e01">

``` python
!ls
!rm coronahack-chest-xraydataset.zip
```

<div class="output stream stdout">

    Chest_xray_Corona_dataset_Summary.csv  kaggle.json
    Chest_xray_Corona_Metadata.csv	       sample_data
    Coronahack-Chest-XRay-Dataset	       test.csv
    coronahack-chest-xraydataset.zip       train.csv

</div>

</div>

<div class="cell markdown" data-colab_type="text" id="cMwK79fQLe0y">

![test
dataset](https://drive.google.com/uc?id=1Iisi74QIWY6CSAYJrf3_lTiSCJOJEq4u)

</div>

<div class="cell markdown" data-colab_type="text" id="RO2EndeaLpkM">

![train
dataset](https://drive.google.com/uc?id=1ORukZs5Z5dJ08HaHj4mCftavwygdjp72)

</div>

<div class="cell markdown" data-colab_type="text" id="Znf7GVZrJVn2">

### Use ImageDataGenerator from keras as it is very helpful for preprocessing the image data for spliting into batches and make varients of them

</div>

<div class="cell code" data-execution_count="0" data-colab="{}" data-colab_type="code" id="EzLhxKftfXv9">

``` python
from keras_preprocessing.image import ImageDataGenerator
```

</div>

<div class="cell code" data-execution_count="0" data-colab="{}" data-colab_type="code" id="hJPgz632UaQ2">

``` python
train=pd.read_csv("train.csv",dtype=str)
test=pd.read_csv("test.csv",dtype=str)
```

</div>

<div class="cell markdown" data-colab_type="text" id="Z9lXSpbYJpwJ">

### Now rescale the image cause we know pixel values are from 0 to 255

</div>

<div class="cell code" data-execution_count="0" data-colab="{}" data-colab_type="code" id="3m7mHJj1V2Q1">

``` python
datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest',validation_split=0.25)
```

</div>

<div class="cell code" data-execution_count="0" data-colab="{}" data-colab_type="code" id="cNzvOvgy9s_W">

``` python
batch_size = 20
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150
```

</div>

<div class="cell markdown" data-colab_type="text" id="oHdkwBNzJ9mf">

Use the flow\_from\_dataframe() method from ImageDataGenerator doc can
be found here
[flow\_from\_dataframe()](https://keras.io/api/preprocessing/image/)

</div>

<div class="cell markdown" data-colab_type="text" id="jzckf41vMDVX">

    ImageDataGenerator.flow_from_dataframe(
        dataframe,
        directory=None,
        x_col="filename",
        y_col="class",
        weight_col=None,
        target_size=(256, 256),
        color_mode="rgb",
        classes=None,
        class_mode="categorical",
        batch_size=32,
        shuffle=True,
        seed=None,
        save_to_dir=None,
        save_prefix="",
        save_format="png",
        subset=None,
        interpolation="nearest",
        validate_filenames=True,
        **kwargs
    )

</div>

<div class="cell code" data-execution_count="0" data-colab="{&quot;height&quot;:52,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" data-colab_type="code" id="C4sx64P8V3I7" data-outputId="334277f9-a5bd-41b3-9cc0-43081f72ce1f">

``` python
train_generator=datagen.flow_from_dataframe(
dataframe=train,
directory="Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train/",
x_col="X_ray_image_name",
y_col="Class",
subset="training",
batch_size=batch_size,
shuffle=True,
seed=32,
target_size=(IMG_HEIGHT, IMG_WIDTH),
class_mode="categorical")

valid_generator=datagen.flow_from_dataframe(
dataframe=train,
directory="Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train/",
x_col="X_ray_image_name",
y_col="Class",
subset="validation",
shuffle=True,
seed=32,
target_size=(IMG_HEIGHT,IMG_WIDTH),
class_mode="categorical")
```

<div class="output stream stdout">

    Found 2059 validated image filenames belonging to 3 classes.
    Found 686 validated image filenames belonging to 3 classes.

</div>

</div>

<div class="cell code" data-execution_count="0" data-colab="{}" data-colab_type="code" id="PrEDoFEQXYQ4">

``` python
test_datagen = ImageDataGenerator(rescale=1. / 255)
```

</div>

<div class="cell code" data-execution_count="0" data-colab="{&quot;height&quot;:35,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" data-colab_type="code" id="sAuz6Yo6V8Ie" data-outputId="bf7832d9-1fae-4ae1-e1ac-8097f084996a">

``` python
test_generator=test_datagen.flow_from_dataframe(
dataframe=test,
directory="Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test/",
target_size=(IMG_HEIGHT,IMG_WIDTH),
x_col="X_ray_image_name",
batch_size=batch_size,
shuffle=False,
seed=32,
class_mode=None)
```

<div class="output stream stdout">

    Found 382 validated image filenames.

</div>

</div>

<div class="cell markdown" data-colab_type="text" id="bnKtiDHYMlVA">

## Now we import tensorflow layers and models

</div>

<div class="cell code" data-execution_count="0" data-colab="{}" data-colab_type="code" id="jINkyl6KX0FH">

``` python
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
```

</div>

<div class="cell code" data-execution_count="0" data-colab="{}" data-colab_type="code" id="kay1MxFD_2rZ">

``` python
sample_training_images, _ = next(train_generator)
```

</div>

<div class="cell markdown" data-colab_type="text" id="shyTYFcZMtg-">

## We plot the sample images using matplotlib

</div>

<div class="cell code" data-execution_count="0" data-colab="{&quot;height&quot;:306,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" data-colab_type="code" id="AEwtD8xQALZk" data-outputId="be226f56-d22a-43b5-894d-6e7fc193946e">

``` python
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
plotImages(sample_training_images[:5])
```

<div class="output display_data">

![](f9076b1d83e2d1c1d535d84f7fae79f7041552af.png)

</div>

</div>

<div class="cell code" data-execution_count="0" data-colab="{&quot;height&quot;:35,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" data-colab_type="code" id="oc9A8zzHAgJm" data-outputId="1bca62d4-94b3-4873-db3f-27eddbe6fb60">

``` python
sample_training_images[1].shape
```

<div class="output execute_result" data-execution_count="77">

    (150, 150, 3)

</div>

</div>

<div class="cell code" data-execution_count="0" data-colab="{}" data-colab_type="code" id="xOhdu12CZfar">

``` python
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
```

</div>

<div class="cell markdown" data-colab_type="text" id="tKszK7pvM2ku">

The Model is now created as below

</div>

<div class="cell code" data-execution_count="0" data-colab="{}" data-colab_type="code" id="q5tWVQUBX3tC">

``` python
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),padding="valid"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu',padding="valid"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu',padding="valid"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(256,activation="relu"))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(3,activation='softmax'))
```

</div>

<div class="cell code" data-execution_count="0" data-colab="{&quot;height&quot;:515,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" data-colab_type="code" id="Q3-MndaJBGKB" data-outputId="e98d63e2-055f-4511-eda7-ef881b3f7ebd">

``` python
model.summary()
```

<div class="output stream stdout">

    Model: "sequential_2"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_6 (Conv2D)            (None, 148, 148, 32)      896       
    _________________________________________________________________
    max_pooling2d_6 (MaxPooling2 (None, 74, 74, 32)        0         
    _________________________________________________________________
    conv2d_7 (Conv2D)            (None, 72, 72, 32)        9248      
    _________________________________________________________________
    max_pooling2d_7 (MaxPooling2 (None, 36, 36, 32)        0         
    _________________________________________________________________
    conv2d_8 (Conv2D)            (None, 34, 34, 64)        18496     
    _________________________________________________________________
    max_pooling2d_8 (MaxPooling2 (None, 17, 17, 64)        0         
    _________________________________________________________________
    flatten_2 (Flatten)          (None, 18496)             0         
    _________________________________________________________________
    dense_4 (Dense)              (None, 256)               4735232   
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 256)               0         
    _________________________________________________________________
    dense_5 (Dense)              (None, 3)                 771       
    =================================================================
    Total params: 4,764,643
    Trainable params: 4,764,643
    Non-trainable params: 0
    _________________________________________________________________

</div>

</div>

<div class="cell code" data-execution_count="0" data-colab="{&quot;height&quot;:550,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" data-colab_type="code" id="9_qBpvxtBKd_" data-outputId="5386aec3-baec-4678-a00d-2951b2d86c68">

``` python
import tensorflow as tf
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=STEP_SIZE_TRAIN,
    epochs=epochs,
    validation_data=valid_generator,
    validation_steps=STEP_SIZE_VALID
)
```

<div class="output stream stdout">

    Epoch 1/15
    102/102 [==============================] - 50s 488ms/step - loss: 0.7432 - accuracy: 0.6538 - val_loss: 1.3526 - val_accuracy: 0.1949
    Epoch 2/15
    102/102 [==============================] - 50s 486ms/step - loss: 0.6057 - accuracy: 0.7353 - val_loss: 1.0914 - val_accuracy: 0.1473
    Epoch 3/15
    102/102 [==============================] - 50s 488ms/step - loss: 0.4993 - accuracy: 0.7979 - val_loss: 0.5632 - val_accuracy: 0.7426
    Epoch 4/15
    102/102 [==============================] - 49s 480ms/step - loss: 0.4496 - accuracy: 0.8190 - val_loss: 0.8519 - val_accuracy: 0.5565
    Epoch 5/15
    102/102 [==============================] - 49s 481ms/step - loss: 0.4144 - accuracy: 0.8352 - val_loss: 0.5459 - val_accuracy: 0.7500
    Epoch 6/15
    102/102 [==============================] - 49s 481ms/step - loss: 0.4254 - accuracy: 0.8485 - val_loss: 0.5300 - val_accuracy: 0.7738
    Epoch 7/15
    102/102 [==============================] - 50s 489ms/step - loss: 0.3941 - accuracy: 0.8514 - val_loss: 0.3105 - val_accuracy: 0.9018
    Epoch 8/15
    102/102 [==============================] - 49s 481ms/step - loss: 0.3743 - accuracy: 0.8548 - val_loss: 0.2634 - val_accuracy: 0.9241
    Epoch 9/15
    102/102 [==============================] - 50s 488ms/step - loss: 0.3720 - accuracy: 0.8592 - val_loss: 0.1268 - val_accuracy: 0.9851
    Epoch 10/15
    102/102 [==============================] - 49s 484ms/step - loss: 0.3587 - accuracy: 0.8627 - val_loss: 0.3251 - val_accuracy: 0.9226
    Epoch 11/15
    102/102 [==============================] - 50s 486ms/step - loss: 0.3601 - accuracy: 0.8538 - val_loss: 0.2828 - val_accuracy: 0.9226
    Epoch 12/15
    102/102 [==============================] - 49s 480ms/step - loss: 0.3325 - accuracy: 0.8637 - val_loss: 0.3183 - val_accuracy: 0.8899
    Epoch 13/15
    102/102 [==============================] - 50s 491ms/step - loss: 0.3005 - accuracy: 0.8789 - val_loss: 0.2486 - val_accuracy: 0.9390
    Epoch 14/15
    102/102 [==============================] - 51s 498ms/step - loss: 0.3183 - accuracy: 0.8710 - val_loss: 0.5745 - val_accuracy: 0.7336
    Epoch 15/15
    102/102 [==============================] - 50s 490ms/step - loss: 0.2858 - accuracy: 0.8803 - val_loss: 0.1115 - val_accuracy: 0.9955

</div>

</div>

<div class="cell code" data-execution_count="0" data-colab="{&quot;height&quot;:35,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" data-colab_type="code" id="N3IgpbDnZ46m" data-outputId="9c9cff5e-5d1e-4c3c-e78e-4598dc2c7914">

``` python
test_loss, test_acc=model.evaluate_generator(generator=valid_generator,steps=STEP_SIZE_TEST,verbose=2)
```

<div class="output stream stdout">

    19/19 - 14s - loss: 0.1168 - accuracy: 0.9918

</div>

</div>

<div class="cell markdown" data-colab_type="text" id="8NtJvkWCM-Il">

## We get 99.18% accuracy from the train set now we plot the training and validation loss and accuracy as below

</div>

<div class="cell code" data-execution_count="0" data-colab="{&quot;height&quot;:336,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" data-colab_type="code" id="q-oiOMu0RDfo" data-outputId="18fe3203-50fd-4131-bb51-03ec2a51ada8">

``` python
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
```

<div class="output display_data">

![](f050dd0f872a6f9e0f8fe18d1ee29c00587136ba.png)

</div>

</div>

<div class="cell code" data-execution_count="0" data-colab="{&quot;height&quot;:35,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" data-colab_type="code" id="fXBZv2e9l7H0" data-outputId="6c5743be-b1e7-43f5-ee6f-7b9c8f101828">

``` python
print(test_acc)
```

<div class="output stream stdout">

    0.9917762875556946

</div>

</div>

<div class="cell markdown" data-colab_type="text" id="eFRSfWlfNWVX">

# So our test accuracy of the model is 99.17%

</div>
