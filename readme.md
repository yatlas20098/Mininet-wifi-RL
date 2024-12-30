#### Install Mininet-Wifi:

**We highly recommend using Ubuntu version 16.04 or higher. Some new hostapd features might not work on Ubuntu 14.04.**
step 1:

```
sudo apt-get install git
```

step 2: 

```
git clone https://github.com/intrig-unicamp/mininet-wifi
```

step 3: 

```
cd mininet-wifi
```

step 4: 

```
sudo util/install.sh -Wlnfv
```

##### install.sh options:

-W: wireless dependencies
-n: mininet-wifi dependencies
-f: OpenFlow
-v: OpenvSwitch
-l: wmediumd
*optional*:
-P: P4 dependencies
-6: wpan tools

#### Install RL:

```
pip install -r requirements.txt
```

#### Run the code:

```
sudo -E python3 simulation.py
```

#### Clean mininet if the program did not exit regularly (crashed or terminated):
```
sudo mn -c
```

(reminder: this repository does not include dataset. Require to adjust path to run the code.)

#### Analyze the result:

```
sudo chmod -R 777 /output
```

Analyze the pacp file:

```
./extract_pacp.sh
```

Draw the timeseries plot of throughput for each sensor and one for all the sensors:

```
python3 plot_timeseries.py
```



### Mininet-WIFI introduction:

Set up network:

Wmediumd is a simulate interface to crate realistic wireless network.

```
net = Mininet_wifi(controller=Controller, link=wmediumd,wmediumd_mode=interference)
```

Set up accesspoint:

mode=g means WIFI 802.11g, and set position at 50,50,0

```
ap23 = net.addAccessPoint('ap23', ssid='new-ssid', mode='g', channel='5', position='50,50,0')
```

Set up sensor:

We can set up ip address, communication range, and position for the sensor. The signal strength will be influenced by the distance between accesspoint and sensor. 

```
net.addStation('sensor1', ip=192.168.0.1/24,range='116', position='30,30')
```

Set up controller:

```
net.addController('c0')
```

Configuring WIFI nodes:

```
net.configureWifiNodes()
```

Start WIFI:

```
net.build()
net.start()
```

Running CLI:

We can do a lot of thing in CLI, for example, we can use ping command to ping each sensors.

```
CLI(net)
```

Stop network:

```
net.stop()
```

More example can be found here: https://github.com/intrig-unicamp/mininet-wifi/tree/master/examples

#### Cloudlab Introduction:

To get access of Cloudlab node, the first step is to put SSH key of your local machine. You can find Manage SSH Keys in the top right, under your name.

Then we have to reserve a node: Click Experiments in the top right, and then click Reserve Node.

Different Cluster contain different configuration of node, you can choose based on availability and requirements. (For this experiment, what I usually use is c240g5 under Wisconsin Cluster).
![image-20240911023320253](https://github.com/user-attachments/assets/e5565493-d773-4172-9a1a-86e19d1e4159)


After we reserve a node, and our reservation is start, we can start to create a experiments:

Profile is used to preload dataset, and version of OS. The Small-lan:38 is the default set up, which will give you a plain node.

![image-20240911023744001](https://github.com/user-attachments/assets/26f88cec-2288-4be9-980e-c5d2309aa16d)


Next, you can select the OS image you want, and choose the node type(the node you reserve). Since the disk space under user is very small, I suggest to check box **Temp Filesystem Max Space**. It will give you a lot of space under /mydata.

![image-20240911023919053](https://github.com/user-attachments/assets/6234b244-ab00-45fe-84ad-a888237b2d51)

The last step is choose the start time. If you do not choose, your machine will be start immediately. For the first use machine, the experiment can only use for 16 hours, but you can extend twice before expire.

![image-20240911024211526](https://github.com/user-attachments/assets/6b50cfba-534b-499f-bbeb-8e8cf451f7ec)

Once you start the experiment, you can use the node you reserve.

This is the portal for your experiment. What we have to do is to use the link under SSH command to connect with node, for example, we can use VScode Remote-ssh, mobaxterm, and anything else.

 the Extend button can extend experiment for at least 14 days.

![image-20240911024400389](https://github.com/user-attachments/assets/89e39614-5bc3-4683-b186-33be53a42385)

Under Topology View, we can choose to restart the node or reload the node.

![image-20240911024747633](https://github.com/user-attachments/assets/9959eaf3-f40e-473a-b7cc-ffd2e5ee4a75)

