# CapstoneDesign
인공지능캡스톤디자인 NeRFBot :TurtleBot3를 활용한 NeRF 데이터셋 취득

소프트웨어융합대학 캡스톤 Fair 3rd Place 우수상

![캡스톤_발표포스터_최종본_1](https://github.com/qkrwnsdn0427/CapstoneDesign/assets/129582433/878c6416-ca68-45b2-87ff-a3e7c1a562cd)




- Instant NeRF(Linux)
    
    **Building instant-ngp (Windows & Linux)**
    
    **Requirements**
    
    - An **NVIDIA GPU**; tensor cores increase performance when available. All shown results come from an RTX 3090.
    - A **C++14** capable compiler. The following choices are recommended and have been tested:
        - **Windows:** Visual Studio 2019 or 2022
        - **Linux:** GCC/G++ 8 or higher
    - A recent version of [**CUDA**](https://developer.nvidia.com/cuda-toolkit). The following choices are recommended and have been tested:
        - **Windows:** CUDA 11.5 or higher
        - **Linux:** CUDA 10.2 or higher
    - [**CMake](https://cmake.org/) v3.21 or higher**.
    - **(optional) [Python](https://www.python.org/) 3.7 or higher** for interactive bindings. Also, run `pip install -r requirements.txt`.
    - **(optional) [OptiX](https://developer.nvidia.com/optix) 7.6 or higher** for faster mesh SDF training.
    - **(optional) [Vulkan SDK](https://vulkan.lunarg.com/)** for DLSS support.
    
    구현환경
    
    Ubuntu 20.04
    
    GPU : RTX2060
    
    cmake 3.21 이상
    
    NVIDIA 그래픽 드라이버 설치
    
    ```jsx
    sudo ubuntu-drivers devices // 설치가 가능한 ubuntu 설치 드라이버 목록이 나옴
    sudo apt install nvdia-driver-470 //예시
    sudo reboot
    
    //재부팅 이후
    nvdia-smi//로 확인
    ```
    
    NVIDIA CUDA Toolkit 설치
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/c3a3eddb-3deb-43f5-b9db-57491f74a674/aa7255a3-075b-4409-8ad1-211d9c588943/Untitled.png)
    
    …
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/c3a3eddb-3deb-43f5-b9db-57491f74a674/a2fad324-2bf3-4229-9bc4-a3f8db400818/Untitled.png)
    
    [Wikiwand - CUDA](https://www.wikiwand.com/en/CUDA#/GPUs_supported)
    
    위 목록에서 자신의 GPU에 맞는 CUDA 버전 확인
    
    ex) RTX2060 → 7.5 Turing
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/c3a3eddb-3deb-43f5-b9db-57491f74a674/35d03044-f893-4969-93ec-592adaf447f0/Untitled.png)
    
    [GPU에 맞는 CUDA 버전 확인](https://velog.io/@openjr/GPU에-맞는-CUDA-버전-확인)
    
    초록색의 범위가 호환 가능한 범위
    
    RTX2060 이 7.5 Turing이므로 CUDA 10.0-10.2~12.0까지 설치 가능
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/c3a3eddb-3deb-43f5-b9db-57491f74a674/c4e9d489-389c-4583-a044-ada74fee506d/Untitled.png)
    
    예시로 11.7 설치 
    
    ```jsx
    wget https://developer.download.nvidia.com/compute/cuda/11.7.1/local_installers/cuda_11.7.1_515.65.01_linux.run
    sudo sh cuda_11.7.1_515.65.01_linux.run
    ```
    
    [CUDA Toolkit 11.7 Update 1 Downloads](https://developer.nvidia.com/cuda-11-7-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=runfile_local)
    
    이후 환경변수 설정
    
    ```jsx
    export PATH=/usr/local/cuda-11.7/bin${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
    ```
    
    NVIDIA Optix SDK 설치 (option) 빠른 mesh SDF 학습을 위해 사용
    
    [NVIDIA OptiX™ Downloads](https://developer.nvidia.com/designworks/optix/download)
    
    환경변수 설정
    
    ```jsx
    export OptiX_INSTALL_DIR=/home/사용자이름/NVIDIA-OptiX-SDK-7.6.0-linux-x86_64
    ```
    
    필요 패키지 설치
    
    ```jsx
    sudo apt-get install build-essential git python3-dev python3-pip libopenexr-dev libxi-dev libglfw3-dev libglew-dev libomp-dev libxinerama-dev libxcursor-dev
    apt-get install libssl-dev
    ```
    
    Cmake 설치(v3.21이상 필요)
    
    - apt-get으로 설치시 3.10인가? 까지만 되어 버전이 안 맞으므로
    - 아래 홈페이지에서 다운 받아 직접 빌드하여 설치
    
    [CMake - Upgrade Your Software Build System](https://cmake.org/)
    
    [[개발 환경] CMake 최신 버전 설치하기](https://growingdev.blog/entry/개발-환경-CMake-최신-버전-설치하기)
    
    [CMAKE 업그레이드 하기](https://kyubot.tistory.com/144)
    
    참조
    
    ```jsx
    ./bootstrap
    make -j16
    sudo make install
    cmake --version
    ```
    
    Instant-NGP git clone
    
    ```jsx
    git clone --recursive https://github.com/NVlabs/instant-ngp
    ```
    
    CMake로 빌드
    
    ```jsx
    cmake . -B build
    cmake --build build --config RelWithDebInfo -j
    ```
    
    기본 fox데이터 튜토리얼 실행
    
    ```jsx
    ./instant-ngp data/nerf/fox //instant-ngp 디렉토리로 이동 후 fox데이터 실행
    ```
    
    다른 Demo dataset 다운로드 링크
    
    [nerf_synthetic - Google Drive](https://drive.google.com/drive/folders/1JDdLGDruGNXWnM1eqY1FNL9PlStjaKWi)
    
    ## **커스텀 데이터셋로 NeRF 이용해보기(Colmap 라이브러리 필요)**
    
    Colmap 라이브러리 설치
    
    [Installation — COLMAP 3.9-dev documentation](https://colmap.github.io/install.html)
    
    핸드폰으로 촬영한 동영상 instant-ngp 디렉토리에 위치시킨 후
    
    ```jsx
    ./scripts/colmap2nerf.py --video_in ./동영상이름.mp4 --video_fps 초당프레임추출숫자ex)10 --run_colmap --aabb_scale 16
    ```
    
    생성된 images폴더와 transforms.json파일 새object name 폴더에 넣고 data/nerf/아래에 위치
    
    이후 실행
    
    ```jsx
    ./instant-ngp data/nerf/데이터이름
    ```
    
    만약 아래와 같은 ERROR문 발생시
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/c3a3eddb-3deb-43f5-b9db-57491f74a674/b6ac4511-b451-4b1a-8682-5c4c3be239ad/Untitled.jpeg)
    
    데이터 폴더 크기 비교 시 demo 데이터보다 작음에도 메모리 할당에 실패
    
    새 터미널 열어서 nvidia-smi -l 로 GPU의 메모리 확인하여 메모리가 원인인지 파악하기
    
    → 각 이미지의 해상도가 높아서 에러 발생
    
    → 카메라 화질을 FHD정도로 낮추고 촬영하여 다시 시도해볼 것
