m.AccelerationSensorEnabled=1;
m.AngularVelocitySensorEnabled=1;
m.MagneticSensorEnabled=0;
m.OrientationSensorEnabled=0;
m.PositionSensorEnabled=0;

m.Logging=1;
Time_Past=1;
while(m.Logging)
	[Attitude, Time] = accellog(m);
    [Attitudegyo, Timegyo] = angvellog(m);
	[Attitude_Row,Attitude_Column] = size(Attitude);
    [Attitude_Rowgyo,Attitude_Columngyo] = size(Attitudegyo);
	[Time_Row,Time_Column] = size(Time);
	if Time_Row>0
		if Time(Time_Row,1)>=0
			x=Attitude(Attitude_Row,1);
			y=Attitude(Attitude_Row,2);
			z=Attitude(Attitude_Row,3);
            x_gyo =Attitudegyo(Attitude_Rowgyo,1);
            y_gyo =Attitudegyo(Attitude_Rowgyo,2);
            z_gyo =Attitudegyo(Attitude_Rowgyo,3);
			Time_Now=Time(Time_Row,1);
			if	Time_Past~=Time_Now
			%%	figure('name','Attitude angle')
				subplot(3,2,1)
				plot(Time_Now,x,'*b');
				grid on
				hold on
				xlabel('Time(s)');ylabel('x_acc');
				legend('X');

				subplot(3,2,3)
				plot(Time_Now,y,'*r');
				grid on
				hold on
				xlabel('Time(s)');ylabel('y_acc');
				legend('Y');	
    			

				subplot(3,2,5)
				plot(Time_Now,z,'*g');
				grid on
				hold on
				xlabel('Time(s)');ylabel('z_acc');
				legend('Z');
              

                subplot(3,2,2)
				plot(Time_Now,x_gyo,'*b');
				grid on
				hold on
				xlabel('Time(s)');ylabel('x_gyo');
				legend('X');				
				subplot(3,2,4)
				plot(Time_Now,y_gyo,'*r');
				grid on
				hold on
				xlabel('Time(s)');ylabel('y_gyo');
				legend('Y');				
				subplot(3,2,6)
				plot(Time_Now,z_gyo,'*g');
				grid on
				hold on
				xlabel('Time(s)');ylabel('z_gyo');
				legend('Z');
				Time_Past=Time_Now;
				drawnow;							
			end
		end
	end
end
acc_history=Attitude;
gyo_history=Attitude_Rowgyo;
x_history=Attitude(:,1);
y_history=Attitude(:,2);
z_history=Attitude(:,3);

x = zscore(x_history);
y = zscore(y_history);
z = zscore(z_history);
[bb,ab]=butter(4,0.3);
x1 = filter(bb,ab,x);
y1 = filter(bb,ab,y);
z1 = filter(bb,ab,z);

az = fopen("az.txt",'w');                
fprintf(az,'%g\t',z);
fclose(az);
tz = fopen("tz.txt",'w');                
fprintf(tz,'%g\t',z1);
fclose(az);

xgyo_history=Attitudegyo(:,1);
ygyo_history=Attitudegyo(:,2);
zgyo_history=Attitudegyo(:,3);
figure('name','Acceleration and Gyroscope Data')
subplot(3,1,1)
plot(Time,x_history,'-b');
grid on
hold on
plot(Time,y_history,'-r')
hold on
plot(Time,z_history,'-g')
xlabel('Time(s)');ylabel('Acceleration');
legend('x','y','z');				
subplot(3,1,2)
plot(Timegyo,xgyo_history,'-b');
grid on
hold on
plot(Timegyo,ygyo_history,'-r');
hold on
plot(Timegyo,zgyo_history,'-g');
xlabel('Time(s)');ylabel('Gyroscope');
legend('x','y','z');
subplot(3,1,3)
plot(Time,x1,'-b');
grid on
hold on
plot(Time,y1,'-r');
hold on
plot(Time,z1,'-g');
