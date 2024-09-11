#### Install Mininet-Wifi:

**We highly recommend using Ubuntu version 16.04 or higher. Some new hostapd features might not work on Ubuntu 14.04.**
step 1:

```
 $ sudo apt-get install git
```

step 2: 

```
$ git clone https://github.com/intrig-unicamp/mininet-wifi
```

step 3: 

```
$ cd mininet-wifi
```

step 4: 

```
$ sudo util/install.sh -Wlnfv
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
pip -r install requirements.txt
```

#### Run the code:

```
sudo python3 simulation.py
```

(reminder: this repository does not include dataset, and have to adjust path first)

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

mode=g is WIFI 802.11g, and set position at 50,50,0

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