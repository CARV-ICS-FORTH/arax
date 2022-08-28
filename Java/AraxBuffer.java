package Arax;
import com.sun.jna.Pointer;
import com.sun.jna.Structure;
import java.util.List;
import java.util.Arrays;
import com.sun.jna.Memory;
import com.sun.jna.Native;

public class AraxBuffer
{
	public long user_buffer_size;
	public Pointer arax_data;

	public AraxBuffer(long size)
	{
		user_buffer_size = size;
		arax_data = AraxInterface.INSTANCE.ARAX_BUFFER(size);
	}

	public void free()
	{
		if(arax_data == null)
			throw new RuntimeException("Attempting to free null AraxBuffer");
		AraxInterface.INSTANCE.arax_data_free(arax_data);
		arax_data = null;
	}

	public enum AraxDataFlags {
		NONE_SYNC(0),
		SHM_SYNC(1),
		REMT_SYNC(2),
		ALL_SYNC(3),
		FREE(4);

		private final int value;

		AraxDataFlags(int value)
		{this.value = value;}
		int getAsInt()
		{return value;}
	}

	public void set(AraxAccelerator accel, byte [] data) {
		Pointer mem = new Memory(data.length);
		mem.write(0,data,0,data.length);
		AraxInterface.INSTANCE.arax_data_set(arax_data,accel.getPointer(),mem);
	}

	public void get(byte [] data) {
		Pointer mem = new Memory(data.length);
		AraxInterface.INSTANCE.arax_data_get(arax_data,mem);
		mem.read(0,data,0,data.length);
	}

	public Pointer getPointer()
	{
		return arax_data;
	}
}
