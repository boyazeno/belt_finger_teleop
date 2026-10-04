FROM osrf/ros:jazzy-desktop-full

SHELL [ "/bin/bash" , "-c" ]

# Upgrade all packages
RUN sudo apt update && sudo apt upgrade -y

# Install essential packages
RUN sudo apt update && sudo apt install -y \
    wget git python3-pip python-is-python3 python3-venv \
    build-essential cmake autoconf libtool pkg-config curl unzip

RUN sudo apt install -y \
    android-tools-adb \
    default-jre \
    usbutils \
    unzip

# Create a venv
ENV VIRTUAL_ENV=/opt/venv

RUN python3 -m venv $VIRTUAL_ENV

ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc && \
    echo "source /opt/venv/bin/activate" >> ~/.bashrc

# Install Deoxys control library
WORKDIR /root/ws/3rd_party_libs
RUN apt remove -y libprotobuf-dev protobuf-compiler || true
RUN git clone https://github.com/UT-Austin-RPL/deoxys_control.git
WORKDIR /root/ws/3rd_party_libs/deoxys_control/deoxys
RUN sed -i 's/franka::Robot robot(robot_ip);/franka::Robot robot(robot_ip,franka::RealtimeConfig::kIgnore);/g' franka-interface/src/franka_control_node.cpp
RUN export OSC_FILE=$(find . -name "osc_impedance.cpp") && \
    python3 -c "import re, os; f=os.environ['OSC_FILE']; c=open(f).read(); c=re.sub(r'static_q_task_\s*<<\s*0\.09017809387254755[^;]+;', 'static_q_task_ << -1.4401,1.6179,1.9178,-2.4664,-1.3233,1.3817,1.1431;', c); open(f,'w').write(c)"
RUN chmod +x InstallPackage && printf "0.9.0\n" | ./InstallPackage
RUN make -j build_deoxys=1
RUN pip3 install --break-system-packages --ignore-installed -U -r requirements.txt
RUN make -j build_franka=1
ENV LD_LIBRARY_PATH=/root/ws/3rd_party_libs/deoxys_control/deoxys/lib:$LD_LIBRARY_PATH
RUN pip3 install pyasyncore
RUN pip3 install -e .

# Install VR dependencies
WORKDIR /tmp
RUN wget https://github.com/Genymobile/gnirehtet/releases/download/v2.5.1/gnirehtet-java-v2.5.1.zip && \
    unzip gnirehtet-java-v2.5.1.zip && \
    mv gnirehtet-java/gnirehtet.jar /usr/local/bin/gnirehtet.jar && \
    mv gnirehtet-java/gnirehtet.apk /usr/local/bin/gnirehtet.apk && \
    rm -rf /tmp/gnirehtet*

# Create a global binary wrapper that handles working-directory paths automatically
RUN printf '#!/bin/bash\ncd /usr/local/bin\nexec java -jar gnirehtet.jar "$@"\n' > /usr/local/bin/gnirehtet && \
    chmod +x /usr/local/bin/gnirehtet
    
# Install teleop library
WORKDIR /root/ws/3rd_party_libs
# Copy project directories into the container image
COPY ikfast_pybind ./ikfast_pybind
COPY panda_control ./panda_control
COPY vr_sim_streamer ./vr_sim_streamer
COPY belt_finger_control ./belt_finger_control
RUN cd ikfast_pybind && \
    mkdir -p build && cd build && \
    cmake .. && \
    make install -j4

WORKDIR /root/ws/3rd_party_libs
RUN pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu129 && \
    pip install --no-cache-dir iopath fvcore
RUN pip install --no-cache-dir --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"
RUN pip install --no-cache-dir \
    -r ./panda_control/requirements.txt \
    -r ./vr_sim_streamer/requirements.txt \
    -r ./belt_finger_control/requirements.txt

RUN pip3 install --no-cache-dir -e ./panda_control && \
    pip3 install --no-cache-dir -e ./vr_sim_streamer && \
    pip3 install --no-cache-dir -e ./belt_finger_control

    
    WORKDIR /root/ws/ros_ws/src
    COPY .devcontainer/packages.repos .
    
    RUN vcs import < packages.repos; \
    cd ..; \
    rosdep install --from-paths src --ignore-src -r -y; \
    source /opt/ros/${ROS_DISTRO}/setup.bash; \
    colcon build;
    
    # Add sourcing ROS setup.bash to .bashrc
    RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc
    
    RUN rm -rf /var/lib/apt/lists/*