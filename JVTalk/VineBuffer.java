package Vinetalk;
import com.sun.jna.Pointer;
import com.sun.jna.Structure;
import java.util.List;
import java.util.Arrays;
import com.sun.jna.Memory;
import com.sun.jna.Native;

public class VineBuffer extends Structure
{
	public Pointer user_buffer;
	public long user_buffer_size;
	public Pointer vine_data;
	private Object juser_buffer;
	private Class juser_class;

	protected List<String> getFieldOrder()
	{
		return Arrays.asList(new String[] {"user_buffer","user_buffer_size","vine_data"});
	}

	public VineBuffer()
	{}

	public VineBuffer(Structure struct)
	{
		this(struct,1,true);	// Synchronize by default
	}

	public VineBuffer(Structure struct, boolean sync)
	{
		this(struct,1,sync);	// Synchronize by default
	}

	public VineBuffer(Structure struct, int elementNo)
	{
		this(struct,elementNo,true);	// Synchronize by default
	}

	public VineBuffer(Structure struct, int elementNo, boolean sync)
	{
		user_buffer = struct.getPointer();
		user_buffer_size = struct.size()*elementNo;
//		juser_buffer = null;
		juser_buffer = struct;
		juser_class = Structure.class;
		if(sync)
			write();
	}

	public VineBuffer(byte [] data)
	{
		this(data,true);	// Synchronize by default
	}

	public VineBuffer(byte [] data, boolean sync)
	{
		Pointer mem = new Memory(data.length);
		mem.write(0,data,0,data.length);
		user_buffer = mem;
		user_buffer_size = data.length;
		juser_buffer = data;
		juser_class = Byte.class;
		if(sync)
			write();
	}

	public VineBuffer(float [] data)
	{
		this(data,true);	// Synchronize by default
	}

	public VineBuffer(float [] data, boolean sync)
	{
		int bytes = data.length*Native.getNativeSize(Float.class);
		Pointer mem = new Memory(bytes);

		mem.write(0, data, 0, data.length);

		user_buffer = mem;
		user_buffer_size = bytes;
		juser_buffer = data;
		juser_class = Float.class;
		if(sync)
			write();
	}

	public VineBuffer(long [] data)
	{
		this(data,true);	// Synchronize by default
	}

	public VineBuffer(long [] data, boolean sync)
	{
		int bytes = data.length*Native.getNativeSize(Long.class);
		Pointer mem = new Memory(bytes);
		mem.write(0, data, 0, data.length);
		user_buffer = mem;
		user_buffer_size = bytes;
		juser_buffer = data;
		juser_class = Long.class;
		if(sync)
			write();
	}

	public VineBuffer(double [] data)
	{
		this(data,true);	// Synchronize by default
	}

	public VineBuffer(double [] data, boolean sync)
	{
		int bytes = data.length*Native.getNativeSize(Double.class);
		Pointer mem = new Memory(bytes);
		mem.write(0, data, 0, data.length);
		user_buffer = mem;
		user_buffer_size = bytes;
		juser_buffer = data;
		juser_class = Double.class;
		if(sync)
			write();
	}

	public VineBuffer(int [] data)
	{
		this(data,true);	// Synchronize by default
	}

	public VineBuffer(int [] data, boolean sync)
	{
		int bytes = data.length*Native.getNativeSize(Integer.class);
		Pointer mem = new Memory(bytes);
		mem.write(0, data, 0, data.length);
		user_buffer = mem;
		user_buffer_size = bytes;
		juser_buffer = data;
		juser_class = Integer.class;
		if(sync)
			write();
	}

	public void copyFrom(VineBuffer source)
	{
		user_buffer = source.user_buffer;
		user_buffer_size = source.user_buffer_size;
		vine_data = source.vine_data;
		juser_buffer = source.juser_buffer;
		juser_class = source.juser_class;
	}

	public void read()
	{
		super.read();
		if(user_buffer == null)
			return;

		if(juser_class == Byte.class)
		{
			byte [] data = user_buffer.getByteArray(0,(int)user_buffer_size);
			if(juser_buffer != null)
				System.arraycopy(data,0,juser_buffer,0,(int)user_buffer_size);
		}
		else
		if(juser_class == Float.class)
		{
			int elements = (int)user_buffer_size/Native.getNativeSize(Float.class);
			float [] data = user_buffer.getFloatArray(0,elements);
			if(juser_buffer != null) {
				System.arraycopy(data,0,juser_buffer,0,elements);
			}
		}
		else
		if(juser_class == Long.class)
		{
			int elements = (int)user_buffer_size/Native.getNativeSize(Long.class);
			long [] data = user_buffer.getLongArray(0,elements);
			if(juser_buffer != null)
				System.arraycopy(data,0,juser_buffer,0,elements);
		}
		else
		if(juser_class == Double.class)
		{
			int elements = (int)user_buffer_size/Native.getNativeSize(Double.class);
			double [] data = user_buffer.getDoubleArray(0,elements);
			if(juser_buffer != null)
				System.arraycopy(data,0,juser_buffer,0,elements);
		}
		else
		if(juser_class == Integer.class)
		{
			int elements = (int)user_buffer_size/Native.getNativeSize(Integer.class);
			int [] data = user_buffer.getIntArray(0,elements);
			if(juser_buffer != null)
				System.arraycopy(data,0,juser_buffer,0,elements);
		}
		else
		if(juser_class == Structure.class)
		{
			if(juser_buffer != null) {
			    Structure data = (Structure) juser_buffer;
			    data.read();
		   }
	    } else {
            assert null!="Invalid juser_class!";
	    }
	}
}
