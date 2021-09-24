package Vinetalk;
import com.sun.jna.Pointer;
import com.sun.jna.Structure;
import java.util.List;
import java.util.Arrays;
import com.sun.jna.Memory;
import com.sun.jna.Native;

public class VineBuffer
{
	public long user_buffer_size;
	public Pointer vine_data;

	public VineBuffer(long size)
	{
		user_buffer_size = size;
		vine_data = VineTalkInterface.INSTANCE.VINE_BUFFER(size);
	}

	public void free()
	{
		if(vine_data == null)
			throw new RuntimeException("Attempting to free null VineBuffer");
		VineTalkInterface.INSTANCE.vine_data_free(vine_data);
		vine_data = null;
	}

	public enum VineDataFlags {
		NONE_SYNC(0),
		SHM_SYNC(1),
		REMT_SYNC(2),
		ALL_SYNC(3),
		FREE(4);

		private final int value;

		VineDataFlags(int value)
		{this.value = value;}
		int getAsInt()
		{return value;}
	}

	public void set(VineAccelerator accel, byte [] data) {
		Pointer mem = new Memory(data.length);
		mem.write(0,data,0,data.length);
		VineTalkInterface.INSTANCE.vine_data_set(vine_data,accel.getPointer(),mem);
	}

	public void get(byte [] data) {
		Pointer mem = new Memory(data.length);
		VineTalkInterface.INSTANCE.vine_data_get(vine_data,mem);
		mem.read(0,data,0,data.length);
	}

	public Pointer getPointer()
	{
		return vine_data;
	}
}
