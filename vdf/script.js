function toggleSlice(btn,slice)
{
	elem = document.getElementById(slice);
	if(elem.className == 'slice0')
	{
		elem.className = 'slice1';
		btn.className = 'btn1';
	}
	else
	{
		elem.className = 'slice0';
		btn.className = 'btn0';
	}
}

function highlight_same(obj)
{
	if(this.prev != null)
	{
		var names = document.getElementsByName(this.prev);
		for(i = 0 ; i < names.length; i++)
		{
			names[i].className = '';
		}
	}
	if(obj.getAttribute('name') != null)
	{
		var names = document.getElementsByName(obj.getAttribute('name'));
		for(i = 0 ; i < names.length; i++)
		{
			if(names.length > 1)
				names[i].className = 'GoodBG';
			else
				names[i].className = 'BadBG';
		}
		this.prev = obj.getAttribute('name');
	}
}

function blockTogle(name)
{
	block = document.getElementsByName(name)[0];
	if(block.className == "block")
		block.className = "block_show";
	else
		block.className = "block";
}

function resortGraph(text)
{
	svg = text.parentNode;
	bars = svg.getElementsByTagName('rect')
	boff = parseFloat(svg.getAttribute('data-boff'))

	if(svg.getAttribute('data-sort') == 'time')
	{
		for(i = 0 ; i < bars.length ; i++)
		{
			new_x = boff*parseFloat(bars[i].getAttribute("hist_id"))
			bars[i].setAttribute('x',new_x)
		}
		svg.setAttribute('data-sort',"CDF");
		svg.getElementById('title').innerHTML = "&#x1f441; Duration";
	}
	else
	{
		for(i = 0 ; i < bars.length ; i++)
		{
			for(i = 0 ; i < bars.length ; i++)
			{
				new_x = boff*parseFloat(bars[i].getAttribute("time_id"))
				bars[i].setAttribute('x',new_x)
			}
			svg.setAttribute('data-sort',"time");
		}
		svg.getElementById('title').innerHTML = "&#x1f441; Start";
	}
}

function barInfo(rect,svg_id,all)
{
	svg = document.getElementById(svg_id)
	info = svg.getElementById('task_stuff');
	info.innerHTML = "Task#"+rect.getAttribute('time_id') +" Percentile:"+rect.getAttribute('hist_id')/all + " Duration:" + rect.getAttribute('duration')
}
