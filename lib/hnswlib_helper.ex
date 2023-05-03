defmodule HNSWLib.Helper do
  @moduledoc false

  def get_keyword!(opts, key, type, default, allow_nil? \\ false) do
    val = opts[key] || default

    if allow_nil? and val == nil do
      val
    else
      case get_keyword(key, opts[key] || default, type) do
        {:ok, val} ->
          val

        {:error, reason} ->
          raise ArgumentError, reason
      end
    end
  end

  defp get_keyword(_key, val, :non_neg_integer) when is_integer(val) and val >= 0 do
    {:ok, val}
  end

  defp get_keyword(key, val, :non_neg_integer) do
    {:error,
     "expect keyword parameter `#{inspect(key)}` to be a non-negative integer, got `#{inspect(val)}`"}
  end

  defp get_keyword(_key, val, :pos_integer) when is_integer(val) and val > 0 do
    {:ok, val}
  end

  defp get_keyword(key, val, :pos_integer) do
    {:error,
     "expect keyword parameter `#{inspect(key)}` to be a positive integer, got `#{inspect(val)}`"}
  end

  defp get_keyword(_key, val, :integer) when is_integer(val) do
    {:ok, val}
  end

  defp get_keyword(key, val, :integer) do
    {:error, "expect keyword parameter `#{inspect(key)}` to be an integer, got `#{inspect(val)}`"}
  end

  defp get_keyword(_key, val, :boolean) when is_boolean(val) do
    {:ok, val}
  end

  defp get_keyword(key, val, :boolean) do
    {:error, "expect keyword parameter `#{inspect(key)}` to be a boolean, got `#{inspect(val)}`"}
  end

  defp get_keyword(_key, val, :function) when is_function(val) do
    {:ok, val}
  end

  defp get_keyword(key, val, :function) do
    {:error, "expect keyword parameter `#{inspect(key)}` to be a function, got `#{inspect(val)}`"}
  end

  defp get_keyword(_key, val, {:function, arity})
       when is_integer(arity) and arity >= 0 and is_function(val, arity) do
    {:ok, val}
  end

  defp get_keyword(key, val, {:function, arity}) when is_integer(arity) and arity >= 0 do
    {:error,
     "expect keyword parameter `#{inspect(key)}` to be a function that can be applied with #{arity} number of arguments , got `#{inspect(val)}`"}
  end

  defp get_keyword(_key, val, :atom) when is_atom(val) do
    {:ok, val}
  end

  defp get_keyword(key, val, {:atom, allowed_atoms})
       when is_atom(val) and is_list(allowed_atoms) do
    if val in allowed_atoms do
      {:ok, val}
    else
      {:error,
       "expect keyword parameter `#{inspect(key)}` to be an atom and is one of `#{inspect(allowed_atoms)}`, got `#{inspect(val)}`"}
    end
  end

  def list_of_binary(data) when is_list(data) do
    count = Enum.count(data)

    if count > 0 do
      first = Enum.at(data, 0)

      if is_binary(first) do
        expected_size = byte_size(first)

        if rem(expected_size, HNSWLib.Nif.float_size()) != 0 do
          raise ArgumentError,
                "vector feature size should be a multiple of #{HNSWLib.Nif.float_size()} (sizeof(float))"
        else
          features = trunc(expected_size / HNSWLib.Nif.float_size())

          if list_of_binary(data, expected_size) == false do
            raise ArgumentError, "all vectors in the input list should have the same size"
          else
            {count, features}
          end
        end
      end
    else
      {0, 0}
    end
  end

  defp list_of_binary([elem | rest], expected_size) when is_binary(elem) do
    if byte_size(elem) == expected_size do
      list_of_binary(rest, expected_size)
    else
      false
    end
  end

  defp list_of_binary([], expected_size) do
    expected_size
  end

  def verify_data_tensor!(self, data = %Nx.Tensor{}) do
    {rows, features} =
      case data.shape do
        {rows, features} ->
          ensure_vector_dimension!(self, features, {rows, features})

        {features} ->
          ensure_vector_dimension!(self, features, {1, features})

        shape ->
          raise ArgumentError,
                "Input vector data wrong shape. Number of dimensions #{tuple_size(shape)}. Data must be a 1D or 2D array."
      end

    {Nx.to_binary(Nx.as_type(data, :f32)), rows, features}
  end

  def ensure_vector_dimension!(%{dim: dim}, dim, ret), do: ret

  def ensure_vector_dimension!(%{dim: dim}, features, _ret) do
    raise ArgumentError,
          "Wrong dimensionality of the vectors, expect `#{dim}`, got `#{features}`"
  end

  def might_be_float_data!(data) do
    if rem(byte_size(data), float_size()) != 0 do
      raise ArgumentError,
            "vector feature size should be a multiple of #{HNSWLib.Nif.float_size()} (sizeof(float))"
    end
  end

  def normalize_ids!(ids = %Nx.Tensor{}) do
    case ids.shape do
      {_} ->
        Nx.to_binary(Nx.as_type(ids, :u64))

      shape ->
        raise ArgumentError, "expect ids to be a 1D array, got `#{inspect(shape)}`"
    end
  end

  def normalize_ids!(ids) when is_list(ids) do
    if Enum.all?(ids, fn x ->
         is_integer(x) and x >= 0
       end) do
      for item <- ids, into: "", do: <<item::unsigned-integer-native-64>>
    else
      raise ArgumentError, "expect `ids` to be a list of non-negative integers"
    end
  end

  def normalize_ids!(nil) do
    <<>>
  end

  def float_size do
    HNSWLib.Nif.float_size()
  end
end
